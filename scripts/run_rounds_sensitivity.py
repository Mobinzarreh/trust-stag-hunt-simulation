#!/usr/bin/env python3
"""
Horizon sensitivity: does the comparison between the trust agent and the benchmarks depend on the number of
rounds K (the "time window")? (Appendix B of the paper; K = 100 everywhere else.)

Design. The four agents (trust, cumulative Bayes, prior-preserving discounted rho = 0.9 = DiscPP, window w = 10, plus
the plain discounted model as the sensitivity row) play every partner for K rounds, K in config.ROUNDS_GRID, in the
study game (u = 2/3) with the default parameters. The trust agent runs in the agent-1 RNG slot and the benchmarks in
the agent-2 slot, as in B1, so trust-minus-benchmark differences are unpaired (Welch 95% CIs). K = 100 must reproduce
the B1 runs bit for bit (checked for the nine manuscript partners).

What stretches with K. Partners that are indexed by time build their schedule from K: Single Cycle collaborates for
the first 30% of the rounds, Gradual Deterioration slides from 1.0 to 0.2 over the whole horizon, Break and Repair
breaks over the 30% to 60% stretch, Reliability Ramp ramps over 20% / 50% / 80% of K. So at a larger K these partners
change character slowly and the agent has more rounds per phase; at a smaller K the phases are short. Every other
partner (stationary, reactive, belief-driven, Distrust Lock, Reliable but Noisy) is the same partner at every K,
and only the agent's experience grows. Hold the schedule fixed and extend the tail instead? Not built; it would be
a different question (what happens after the regime change has settled).

What is comparable across K. Total payoff scales with K, so the payoff is reported per round (Pi / K) and as
efficiency EF = 100 Pi / HB (a ratio of means); SC and FC are rates, already normalized by K. Contrasts are
trust minus benchmark in each of these, per K and partner. The script also lists the cells whose significance sign
differs from the K = 100 verdict.

The partners are the nine manuscript partners plus the four diagnostic ones (the metric panel's 13) by default. K must be
a multiple of 10 with those: Reliability Ramp puts its breakpoints at 20% / 50% / 80% of K and needs equal integer ramps
(K = 25 raises), which is why the grid has 20, not 25. The script checks every partner x K before simulating.

Outputs (in --out, default results/rounds_sensitivity/): per_run.csv, summary.csv, contrasts.csv, sign_changes.csv.

Usage:
    python scripts/run_rounds_sensitivity.py                      # reuse per_run.csv if it matches
    python scripts/run_rounds_sensitivity.py --overwrite          # regenerate
    python scripts/run_rounds_sensitivity.py --rounds 50 100 400  # another grid (K = 100 is still checked against B1)
    python scripts/run_rounds_sensitivity.py --partners manuscript --runs 20 --out results/rounds_smoke --overwrite
"""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from trust_bayesian_agent_comparison.agents import (
    FocalAgent,
    BayesianFocalAgent,
    PriorPreservingDiscountedBetaAgent,
    # the plain model (the published form) is reported beside the prior-preserving one, never chosen between
    DiscountedBetaAgent,
    SlidingWindowBetaAgent,
)
from trust_bayesian_agent_comparison.analysis import run_cells, per_run_metrics, welch_ci, sig_mark
from trust_bayesian_agent_comparison.partners import (
    MANUSCRIPT_PARTNERS,
    DIAGNOSTIC_PARTNERS,
    PARTNER_CATEGORY,
    partner_factory,
)
from trust_bayesian_agent_comparison.config import (
    RESULTS_DIR,
    NUM_MONTE_CARLO_RUNS,
    NUM_ROUNDS,
    MC_BASE_SEED,
    N_JOBS,
    DISCOUNTED_BETA_RHO,
    RECENCY_BENCHMARK_KEY,
    RECENCY_BENCHMARK_LABEL,
    SLIDING_WINDOW_W,
    ROUNDS_GRID,
)

TRUST = "trust"
# Benchmark key (same as the B1 result directory) -> (label, agent constructor); same four as the OAT and the u sweep
BENCHMARKS = {
    "cumulative": ("Cumulative", BayesianFocalAgent),
    RECENCY_BENCHMARK_KEY: (RECENCY_BENCHMARK_LABEL,
                            partial(PriorPreservingDiscountedBetaAgent, rho=DISCOUNTED_BETA_RHO)),
    f"discounted_rho{DISCOUNTED_BETA_RHO:.2f}": (f"Disc rho={DISCOUNTED_BETA_RHO:.2f}",
                                                 partial(DiscountedBetaAgent, rho=DISCOUNTED_BETA_RHO)),
    f"window_w{SLIDING_WINDOW_W}": (f"Window w={SLIDING_WINDOW_W}",
                                    partial(SlidingWindowBetaAgent, window=SLIDING_WINDOW_W)),
}
LABELS = {TRUST: "Trust", **{k: v[0] for k, v in BENCHMARKS.items()}}
CELL_KEYS = ["K", "partner", "agent"]
METRIC_COLS = ["Pi", "HB", "SC", "FC", "n_CC", "n_CI", "n_IC", "n_II"]
B1_DIR = RESULTS_DIR / "benchmarks"
PARTNER_SETS = {"panel": MANUSCRIPT_PARTNERS + DIAGNOSTIC_PARTNERS, "manuscript": MANUSCRIPT_PARTNERS}


def validate_rounds(partners, rounds):
    """Fail before any simulation if a partner cannot be built at some K (Reliability Ramp needs K a multiple of 10)."""
    for K in rounds:
        for p in partners:
            try:
                partner_factory(p, K)()
            except ValueError as e:
                raise SystemExit(f"{p} cannot be built at K = {K}: {e}. Use K as a multiple of 10 or drop the partner.")


def build_cells(partners, rounds, num_runs):
    """One cell per (K, partner, agent). Longest horizons first, so the joblib queue ends on the short cells."""
    cells = {}
    for K in sorted(rounds, reverse=True):
        for p in partners:
            pf = partner_factory(p, K)
            cells[(K, p, TRUST)] = dict(agent_factory=FocalAgent, partner_factory=pf, slot=1,
                                        num_runs=num_runs, num_rounds=K)
            for b, (_, ctor) in BENCHMARKS.items():
                cells[(K, p, b)] = dict(agent_factory=ctor, partner_factory=pf, slot=2,
                                        num_runs=num_runs, num_rounds=K)
    return cells


def run_sweep(partners, rounds, num_runs, out_dir, overwrite, n_jobs):
    path, meta_path = out_dir / "per_run.csv", out_dir / "per_run_meta.json"
    meta = {"num_runs": num_runs, "base_seed": MC_BASE_SEED, "rounds": sorted(rounds), "partners": list(partners),
            "benchmarks": list(BENCHMARKS)}
    if not overwrite and path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()) != meta:
            raise RuntimeError(f"{path} was made with different settings; rerun with --overwrite")
        print(f"Loading cached sweep from {path}")
        return pd.read_csv(path, float_precision="round_trip")
    cells = build_cells(partners, rounds, num_runs)
    print(f"Running {len(cells)} cells x {num_runs} runs ...")
    res = run_cells(cells, n_jobs=n_jobs, verbose=2)
    frames = [df.assign(**dict(zip(CELL_KEYS, key))) for key, df in res.items()]
    per_run = pd.concat(frames, ignore_index=True)
    per_run = per_run[CELL_KEYS + ["run_id"] + [c for c in per_run.columns if c not in CELL_KEYS + ["run_id"]]]
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(path, index=False)
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"Saved {len(per_run)} runs to {path}")
    return per_run


def check_against_b1(per_run, partners, num_runs):
    """K = NUM_ROUNDS must equal the B1 runs exactly (trust: cumulative/agent1; benchmarks: their own agent2 files)."""
    if num_runs != NUM_MONTE_CARLO_RUNS or NUM_ROUNDS not in set(per_run.K):
        print(f"Cross-check against B1 skipped (needs {NUM_MONTE_CARLO_RUNS} runs and K = {NUM_ROUNDS}).")
        return
    checked = 0
    for p in partners:
        if p not in MANUSCRIPT_PARTNERS:
            continue  # B1 holds the nine manuscript partners only
        for agent in (TRUST, *BENCHMARKS):
            f = B1_DIR / ("cumulative" if agent == TRUST else agent) / f"{p}_{'agent1' if agent == TRUST else 'agent2'}.csv"
            ref = per_run_metrics(pd.read_csv(f))
            ours = per_run[(per_run.K == NUM_ROUNDS) & (per_run.partner == p) & (per_run.agent == agent)]
            ours = ours.sort_values("run_id")
            if not all(np.array_equal(ours[c].to_numpy(), ref[c].to_numpy()) for c in ("Pi", "HB", "SC", "FC")):
                raise AssertionError(f"K = {NUM_ROUNDS} {agent} on {p} differs from {f}")
            checked += 1
    print(f"Cross-check: {checked} cells at K = {NUM_ROUNDS} equal the B1 runs exactly.")


def ratio_ci(pi, hb):
    """Ratio of means 100 * mean(Pi) / mean(HB) and its delta-method standard error."""
    pi, hb = np.asarray(pi, float), np.asarray(hb, float)
    r = pi.mean() / hb.mean()
    n = len(pi)
    var = (pi.var(ddof=1) - 2 * r * np.cov(pi, hb)[0, 1] + r ** 2 * hb.var(ddof=1)) / (n * hb.mean() ** 2)
    return 100 * r, 100 * np.sqrt(max(var, 0.0))


def summarize(per_run):
    out = per_run.groupby(CELL_KEYS, sort=False).agg(
        Pi=("Pi", "mean"), Pi_sd=("Pi", "std"), HB=("HB", "mean"), SC=("SC", "mean"), FC=("FC", "mean"),
        t_mean=("t_mean", "mean"), n=("run_id", "count")).reset_index()
    out["Pi_round"] = out.Pi / out.K
    out["EF"] = 100 * out.Pi / out.HB
    out[["SC", "FC"]] *= 100
    return out


def contrasts(per_run, partners, rounds):
    """Trust minus each benchmark, per K and partner: payoff per round, total payoff, EF, SC, FC (Welch 95% CIs)."""
    groups = {k: g.sort_values("run_id") for k, g in per_run.groupby(CELL_KEYS, sort=False)}
    rows = []
    for K in sorted(rounds):
        for p in partners:
            t = groups[(K, p, TRUST)]
            for b in BENCHMARKS:
                m = groups[(K, p, b)]
                row = {"K": K, "partner": p, "category": PARTNER_CATEGORY[p], "benchmark": b,
                       "trust_Pi_round": t.Pi.mean() / K, "bench_Pi_round": m.Pi.mean() / K}
                for metric, scale in (("Pi_round", 1 / K), ("Pi", 1), ("SC", 100), ("FC", 100)):
                    col = "Pi" if metric == "Pi_round" else metric
                    d, lo, hi = welch_ci(t[col], m[col], scale)
                    row.update({f"d{metric}": d, f"d{metric}_lo": lo, f"d{metric}_hi": hi})
                ef_t, se_t = ratio_ci(t.Pi, t.HB)
                ef_m, se_m = ratio_ci(m.Pi, m.HB)
                d, se = ef_t - ef_m, np.hypot(se_t, se_m)
                row.update({"dEF": d, "dEF_lo": d - 1.96 * se, "dEF_hi": d + 1.96 * se})
                rows.append(row)
    return pd.DataFrame(rows)


def sign_changes(con, ref_k=NUM_ROUNDS):
    """Cells whose significance sign differs from the same partner x benchmark x metric at K = ref_k."""
    if ref_k not in set(con.K):
        return pd.DataFrame()
    rows = []
    for metric in ("Pi_round", "SC", "FC", "EF"):
        for (p, b), g in con.groupby(["partner", "benchmark"], sort=False):
            g = g.set_index("K")
            ref = sig_mark(g.loc[ref_k, f"d{metric}_lo"], g.loc[ref_k, f"d{metric}_hi"])
            for K, r in g.iterrows():
                s = sig_mark(r[f"d{metric}_lo"], r[f"d{metric}_hi"])
                if K != ref_k and s != ref:
                    rows.append({"metric": metric, "partner": p, "benchmark": b, "K": K, "ref_sig": ref or "n.s.",
                                 "sig": s or "n.s.", "d": r[f"d{metric}"], "d_ref": g.loc[ref_k, f"d{metric}"]})
    return pd.DataFrame(rows)


def bar(title):
    print(f"\n{'=' * 124}\n{title}\n{'=' * 124}")


def print_contrast_grid(con, partners, rounds, benchmark, metric, fmt="{:+6.2f}"):
    sub = con[con.benchmark == benchmark].set_index(["partner", "K"])
    print(f"{'Partner':22s}" + "".join(f"{'K=' + str(K):>10s}" for K in rounds))
    for p in partners:
        cells = []
        for K in rounds:
            r = sub.loc[(p, K)]
            cells.append(fmt.format(r[f"d{metric}"]) + (sig_mark(r[f"d{metric}_lo"], r[f"d{metric}_hi"]) or " "))
        print(f"{p:22s}" + "".join(f"{c:>10s}" for c in cells))


def print_tables(summ, con, chg, partners, rounds):
    ordered = sorted(rounds)
    bar("TRUST AGENT ALONE across K (means over runs). Pi/K = payoff per round; EF = 100 Pi / HB; SC, FC in pp of rounds")
    for metric, fmt in (("Pi_round", "{:7.2f}"), ("EF", "{:7.1f}"), ("SC", "{:7.1f}"), ("FC", "{:7.1f}")):
        print(f"\n  {metric}")
        sub = summ[summ.agent == TRUST].set_index(["partner", "K"])
        print(f"  {'Partner':22s}" + "".join(f"{'K=' + str(K):>9s}" for K in ordered))
        for p in partners:
            print(f"  {p:22s}" + "".join(f"{fmt.format(sub.loc[(p, K), metric]):>9s}" for K in ordered))

    units = {"Pi_round": ("payoff per round", "{:+6.2f}"), "FC": ("failed collaboration, pp", "{:+6.1f}"),
             "SC": ("successful collaboration, pp", "{:+6.1f}"), "EF": ("efficiency EF, pp", "{:+6.1f}")}
    for b, (label, _) in BENCHMARKS.items():
        bar(f"TRUST MINUS {label.upper()} by K (Welch 95% CI; + / - = CI above / below 0)")
        for metric, (unit, fmt) in units.items():
            print(f"\n  {unit}")
            print_contrast_grid(con, partners, ordered, b, metric, fmt)

    bar(f"Does the verdict at K = {NUM_ROUNDS} survive other horizons? Cells whose significance sign differs from "
        f"K = {NUM_ROUNDS} ('n.s.' = CI contains 0)")
    if chg.empty:
        print("No cell changes.")
        return
    n_total = len(con) * 4
    print(f"{len(chg)} of {n_total} (K x partner x benchmark x metric) cells differ from their K = {NUM_ROUNDS} verdict.")
    key = chg[chg.benchmark == RECENCY_BENCHMARK_KEY]
    print(f"\nAgainst {RECENCY_BENCHMARK_LABEL} ({len(key)} cells):")
    if len(key):
        print(key.round(2).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate runs even if a cached sweep exists")
    parser.add_argument("--runs", type=int, default=NUM_MONTE_CARLO_RUNS, help="Monte Carlo runs per cell")
    parser.add_argument("--rounds", type=int, nargs="+", default=list(ROUNDS_GRID), help="Horizons K to run")
    parser.add_argument("--partners", choices=sorted(PARTNER_SETS), default="panel",
                        help="panel = nine manuscript + four diagnostic partners; manuscript = the nine")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "rounds_sensitivity", help="Output directory")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS, help="Parallel jobs (-1 = all cores)")
    args = parser.parse_args()

    partners = list(PARTNER_SETS[args.partners])
    rounds = sorted(set(args.rounds))
    print(f"Horizons K: {rounds}. Partners: {len(partners)} ({args.partners}), runs: {args.runs}, "
          f"agents: {', '.join(LABELS.values())}")

    validate_rounds(partners, rounds)
    per_run = run_sweep(partners, rounds, args.runs, args.out, args.overwrite, args.n_jobs)
    check_against_b1(per_run, partners, args.runs)
    summ = summarize(per_run)
    con = contrasts(per_run, partners, rounds)
    chg = sign_changes(con)
    print_tables(summ, con, chg, partners, rounds)

    summ.round(5).to_csv(args.out / "summary.csv", index=False)
    con.round(5).to_csv(args.out / "contrasts.csv", index=False)
    chg.round(5).to_csv(args.out / "sign_changes.csv", index=False)
    print(f"\nSaved summary.csv, contrasts.csv and sign_changes.csv to {args.out}")


if __name__ == "__main__":
    main()
