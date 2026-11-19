#!/usr/bin/env python3
"""
One-factor-at-a-time (OAT) sensitivity of the trust agent at 300 runs per setting (Figure 2 and
Section 5.3 of the paper; initialization in Section 5.1 comes from the x_0 and t_0 rows).

Each parameter in config.OAT_DESIGN moves to its low and high value while every other parameter stays
at the baseline (the default FocalAgent). Every trust
variant runs in the agent-1 RNG slot, the slot of the baseline, so run i of every variant shares its
random numbers with run i of the baseline, and differences from the baseline are paired.

theta (the logit inverse temperature) is shared with the benchmarks, so the cumulative, prior-preserving discounted
(rho = 0.9) and window (w = 10) benchmarks also run at every theta in OAT_THETA_BENCHMARK_GRID, in the
agent-2 slot as in run_benchmark_comparison.py. Trust-minus-benchmark differences are unpaired (Welch).

Per parameter, partner and metric (Pi, SC, FC) the script reports the paired effect of the low and high
value against the baseline (mean, SE, 95% CI), the swing (high minus low), and the local gradient
(swing divided by the parameter change).

Checks (overlapping cells must equal earlier runs exactly):
    baseline                          = main trust runs (results/benchmarks/cumulative, agent 1)
    benchmarks at theta = 2           = main benchmark runs (agent 2)
    t_max = 5, 20                     = ceiling runs of run_bounds_init_sensitivity.py
    x_0 = 1/2; t_0 = 1                = initialization runs of run_bounds_init_sensitivity.py

Outputs (in --out, default results/oat_sensitivity/): per_run.csv, effects.csv, theta_benchmarks.csv.
Figure: scripts/make_oat_tornado.py.

Usage:
    python scripts/run_oat_sensitivity.py                  # reuse per_run.csv if it matches
    python scripts/run_oat_sensitivity.py --overwrite      # regenerate
    python scripts/run_oat_sensitivity.py --runs 20 --out results/oat_smoke --overwrite
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
    DiscountedBetaAgent,
    SlidingWindowBetaAgent,
)
from trust_bayesian_agent_comparison.analysis import run_cells, per_run_metrics, paired_ci, welch_ci, sig_mark
from trust_bayesian_agent_comparison.partners import MANUSCRIPT_PARTNERS, PARTNER_CATEGORY, partner_factory
from trust_bayesian_agent_comparison.config import (
    RESULTS_DIR,
    PROJECT_ROOT,
    NUM_MONTE_CARLO_RUNS,
    NUM_ROUNDS,
    MC_BASE_SEED,
    N_JOBS,
    DISCOUNTED_BETA_RHO,
    RECENCY_BENCHMARK_KEY,
    RECENCY_BENCHMARK_LABEL,
    SLIDING_WINDOW_W,
    OAT_DESIGN,
    OAT_THETA_BENCHMARK_GRID,
)

BASE = "baseline"
TRUST = "trust"
BENCHMARKS = {
    "cumulative": ("Cumulative", BayesianFocalAgent),
    RECENCY_BENCHMARK_KEY: (RECENCY_BENCHMARK_LABEL,
                            partial(PriorPreservingDiscountedBetaAgent, rho=DISCOUNTED_BETA_RHO)),
    f"discounted_rho{DISCOUNTED_BETA_RHO:.2f}": (f"Disc rho={DISCOUNTED_BETA_RHO:.2f}",
                                                 partial(DiscountedBetaAgent, rho=DISCOUNTED_BETA_RHO)),
    f"window_w{SLIDING_WINDOW_W}": (f"Window w={SLIDING_WINDOW_W}",
                                    partial(SlidingWindowBetaAgent, window=SLIDING_WINDOW_W)),
}
METRICS = (("Pi", 1), ("SC", 100), ("FC", 100))
METRIC_COLS = ["Pi", "HB", "SC", "FC", "n_CC", "n_CI", "n_IC", "n_II"]
B1_DIR = RESULTS_DIR / "benchmarks"
B4_DIR = RESULTS_DIR / "bounds_init"


def variants():
    """Trust variants: key -> (parameter, level name, value, FocalAgent keyword overrides)."""
    out = {BASE: (BASE, BASE, np.nan, {})}
    for param, (kw, base, low, high) in OAT_DESIGN.items():
        for level, value in (("low", low), ("high", high)):
            if value is not None:
                out[f"{param}_{level}"] = (param, level, value, {kw: value})
    theta_kw = OAT_DESIGN["theta"][0]
    for th in OAT_THETA_BENCHMARK_GRID:
        if th not in (OAT_DESIGN["theta"][1:]):
            out[f"theta_{th:g}"] = ("theta", "extra", th, {theta_kw: th})
    return out


def build_cells(partners, num_runs):
    cells = {}
    for key, (_, _, _, overrides) in variants().items():
        for p in partners:
            cells[(TRUST, key, p)] = dict(agent_factory=partial(FocalAgent, **overrides),
                                          partner_factory=partner_factory(p, NUM_ROUNDS), slot=1, num_runs=num_runs)
    for b, (_, ctor) in BENCHMARKS.items():
        for th in OAT_THETA_BENCHMARK_GRID:
            for p in partners:
                cells[(b, f"theta_{th:g}", p)] = dict(agent_factory=partial(ctor, inv_temp=th),
                                                     partner_factory=partner_factory(p, NUM_ROUNDS), slot=2,
                                                     num_runs=num_runs)
    return cells


def run_sweep(partners, num_runs, out_dir, overwrite, n_jobs):
    path, meta_path = out_dir / "per_run.csv", out_dir / "per_run_meta.json"
    meta = {"num_runs": num_runs, "base_seed": MC_BASE_SEED, "num_rounds": NUM_ROUNDS, "partners": list(partners),
            "design": {k: [v[0], v[1], v[2], v[3]] for k, v in OAT_DESIGN.items()},
            "theta_bench": list(OAT_THETA_BENCHMARK_GRID)}
    if not overwrite and path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()) != meta:
            raise RuntimeError(f"{path} was made with different settings; rerun with --overwrite")
        print(f"Loading cached sweep from {path}")
        return pd.read_csv(path, float_precision="round_trip")
    cells = build_cells(partners, num_runs)
    print(f"Running {len(cells)} cells x {num_runs} runs ...")
    res = run_cells(cells, n_jobs=n_jobs, verbose=2)
    per_run = pd.concat([df.assign(agent=a, variant=v, partner=p) for (a, v, p), df in res.items()],
                        ignore_index=True)
    per_run = per_run[["agent", "variant", "partner", "run_id"] +
                      [c for c in per_run.columns if c not in ("agent", "variant", "partner", "run_id")]]
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(path, index=False)
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"Saved {len(per_run)} runs to {path}")
    return per_run


def cell(per_run, agent, variant, partner):
    g = per_run[(per_run.agent == agent) & (per_run.variant == variant) & (per_run.partner == partner)]
    return g.sort_values("run_id").set_index("run_id")


def check_against_cached(per_run, partners, num_runs):
    if num_runs != NUM_MONTE_CARLO_RUNS:
        print("Cross-checks against B1/B4 skipped (need the full run count).")
        return
    pairs = [((TRUST, BASE), B1_DIR / "cumulative" / "{p}_agent1.csv")]
    pairs += [((b, "theta_2"), B1_DIR / b / "{p}_agent2.csv") for b in BENCHMARKS]
    if "t_max_low" in variants():
        pairs += [((TRUST, "t_max_low"), B4_DIR / "ceiling" / "tmax5" / "{p}_agent1.csv"),
                  ((TRUST, "t_max_high"), B4_DIR / "ceiling" / "tmax20" / "{p}_agent1.csv")]
    pairs += [((TRUST, "x_0_high"), B4_DIR / "init" / "x0.500_t0" / "{p}_agent1.csv"),
              ((TRUST, "t_0_high"), B4_DIR / "init" / "x0.333_t1" / "{p}_agent1.csv")]
    checked = 0
    for (agent, variant), pattern in pairs:
        for p in partners:
            f = Path(str(pattern).format(p=p))
            if not f.exists():
                raise FileNotFoundError(f"Cross-check file missing: {f}")
            ref = per_run_metrics(pd.read_csv(f))
            ours = cell(per_run, agent, variant, p)
            if not all(np.array_equal(ours[c].to_numpy(), ref[c].to_numpy()) for c in ("Pi", "HB", "SC", "FC")):
                raise AssertionError(f"{agent}/{variant} on {p} differs from {f}")
            checked += 1
    print(f"Cross-check: {checked} cells equal the B1/B4 runs exactly (baseline, benchmarks at theta = 2, "
          f"{'t_max = 5/20, ' if 't_max_low' in variants() else ''}x_0 = 1/2, t_0 = 1).")




def effects(per_run, partners):
    rows = []
    for param, (kw, base_value, low, high) in OAT_DESIGN.items():
        for p in partners:
            base = cell(per_run, TRUST, BASE, p)
            lv = {lvl: cell(per_run, TRUST, f"{param}_{lvl}", p) for lvl, v in (("low", low), ("high", high))
                  if v is not None}
            for metric, scale in METRICS:
                row = {"parameter": param, "keyword": kw, "baseline_value": base_value, "low_value": low,
                       "high_value": high, "partner": p, "category": PARTNER_CATEGORY[p], "metric": metric,
                       "baseline_mean": scale * base[metric].mean()}
                for lvl, df in lv.items():
                    d, lo, hi = paired_ci(df[metric], base[metric], scale)
                    row.update({f"{lvl}_effect": d, f"{lvl}_lo": lo, f"{lvl}_hi": hi,
                                f"{lvl}_se": (hi - lo) / (2 * 1.96)})
                if "low" in lv:
                    d, lo, hi = paired_ci(lv["high"][metric], lv["low"][metric], scale)
                    span = high - low
                else:
                    d, lo, hi = row["high_effect"], row["high_lo"], row["high_hi"]
                    span = high - base_value
                row.update({"swing": d, "swing_lo": lo, "swing_hi": hi, "gradient": d / span,
                            "gradient_lo": lo / span, "gradient_hi": hi / span})
                rows.append(row)
    return pd.DataFrame(rows)


def theta_benchmarks(per_run, partners):
    rows = []
    for th in OAT_THETA_BENCHMARK_GRID:
        tv = BASE if th == OAT_DESIGN["theta"][1] else (
            "theta_low" if th == OAT_DESIGN["theta"][2] else "theta_high" if th == OAT_DESIGN["theta"][3]
            else f"theta_{th:g}")
        for p in partners:
            t = cell(per_run, TRUST, tv, p)
            for b in BENCHMARKS:
                m = cell(per_run, b, f"theta_{th:g}", p)
                row = {"theta": th, "partner": p, "benchmark": b, "trust_Pi": t.Pi.mean(), "bench_Pi": m.Pi.mean()}
                for metric, scale in METRICS:
                    d, lo, hi = welch_ci(t[metric], m[metric], scale)
                    row.update({f"d{metric}": d, f"d{metric}_lo": lo, f"d{metric}_hi": hi})
                rows.append(row)
    return pd.DataFrame(rows)


def ordering_robustness(per_run, partners):
    """
    Trust minus benchmark (total payoff, Welch) under every OAT variant. The benchmarks do not use the
    trust parameters, so each variant is compared with the benchmark at the same theta (theta = 2 unless
    theta itself is varied). A conclusion "changes" when the significance sign differs from the baseline.
    """
    rows = []
    theta_of = {"theta_low": OAT_DESIGN["theta"][2], "theta_high": OAT_DESIGN["theta"][3]}
    for key, (param, level, value, _) in variants().items():
        if level == "extra":
            continue
        th = theta_of.get(key, OAT_DESIGN["theta"][1])
        for p in partners:
            t = cell(per_run, TRUST, key, p)
            for b in BENCHMARKS:
                m = cell(per_run, b, f"theta_{th:g}", p)
                d, lo, hi = welch_ci(t.Pi, m.Pi)
                rows.append({"variant": key, "parameter": param, "level": level, "value": value, "partner": p,
                             "benchmark": b, "dPi": d, "dPi_lo": lo, "dPi_hi": hi, "sig": sig_mark(lo, hi)})
    out = pd.DataFrame(rows)
    base = out[out.variant == BASE].set_index(["partner", "benchmark"]).sig
    out["baseline_sig"] = [base[(r.partner, r.benchmark)] for r in out.itertuples()]
    out["changed"] = out.sig != out.baseline_sig
    return out


def print_ordering(order, partners):
    bar("Does the trust-vs-benchmark ordering survive every OAT variant? (total payoff, Welch 95% CI)\n"
        "Listed: cells whose significance sign differs from the baseline ('' = n.s.)")
    ch = order[order.changed & (order.variant != BASE)]
    n = len(order[order.variant != BASE])
    print(f"{len(ch)} of {n} variant x partner x benchmark cells change; by parameter:")
    print(ch.groupby("parameter").size().reindex(OAT_DESIGN.keys(), fill_value=0).to_string())
    if len(ch):
        cols = ["variant", "partner", "benchmark", "baseline_sig", "sig", "dPi", "dPi_lo", "dPi_hi"]
        print(ch[cols].replace("", "n.s.").round(1).to_string(index=False))


def bar(title):
    print(f"\n{'=' * 124}\n{title}\n{'=' * 124}")


def fmt_value(v):
    return f"{v:.3g}" if v is not None and np.isfinite(v) else "-"


def print_tables(eff, theta_tab, partners):
    short = {p: p[:11] for p in partners}
    for metric, unit in (("Pi", "total payoff"), ("FC", "failed collaboration, pp"), ("SC", "successful collaboration, pp")):
        bar(f"OAT effects on {unit}: variant minus baseline, paired (+/- = 95% CI above/below 0)")
        print(f"{'parameter (value)':26s}" + "".join(f"{short[p]:>12s}" for p in partners))
        base = eff[(eff.metric == metric) & (eff.parameter == "theta")].set_index("partner").baseline_mean
        print(f"{'baseline mean':26s}" + "".join(f"{base[p]:12.1f}" for p in partners))
        for param, (kw, bval, low, high) in OAT_DESIGN.items():
            sub = eff[(eff.parameter == param) & (eff.metric == metric)].set_index("partner")
            for lvl, val in (("low", low), ("high", high)):
                if val is None:
                    continue
                cells = [f"{sub.loc[p, f'{lvl}_effect']:+.1f}{sig_mark(sub.loc[p, f'{lvl}_lo'], sub.loc[p, f'{lvl}_hi']) or ' '}"
                         for p in partners]
                print(f"{param + ' ' + lvl + ' (' + fmt_value(val) + ')':26s}" + "".join(f"{c:>12s}" for c in cells))

    bar("Ranking by influence on total payoff: largest |swing| over partners, and partners with a significant swing")
    s = eff[eff.metric == "Pi"].copy()
    s["abs_swing"] = s.swing.abs()
    s["sig"] = (s.swing_lo > 0) | (s.swing_hi < 0)
    rank = s.groupby("parameter").agg(max_abs_swing=("abs_swing", "max"), mean_abs_swing=("abs_swing", "mean"),
                                      n_sig=("sig", "sum")).sort_values("mean_abs_swing", ascending=False)
    worst = s.loc[s.groupby("parameter").abs_swing.idxmax()].set_index("parameter").partner
    print(f"{'parameter':12s}{'mean |swing|':>14s}{'max |swing|':>13s}  {'(partner)':24s}{'sig. partners':>14s}")
    for param, r in rank.iterrows():
        print(f"{param:12s}{r.mean_abs_swing:14.2f}{r.max_abs_swing:13.2f}  {worst[param]:24s}{int(r.n_sig):>10d}/9")

    bar("theta, shared with the benchmarks: trust minus benchmark, total payoff (Welch 95% CI)")
    for b, (label, _) in BENCHMARKS.items():
        print(f"\n  vs {label}")
        print(f"  {'Partner':22s}" + "".join(f"{'theta=' + format(t, 'g'):>12s}" for t in OAT_THETA_BENCHMARK_GRID))
        sub = theta_tab[theta_tab.benchmark == b].set_index(["partner", "theta"])
        for p in partners:
            cells = [f"{sub.loc[(p, t), 'dPi']:+.1f}{sig_mark(sub.loc[(p, t), 'dPi_lo'], sub.loc[(p, t), 'dPi_hi']) or ' '}"
                     for t in OAT_THETA_BENCHMARK_GRID]
            print(f"  {p:22s}" + "".join(f"{c:>12s}" for c in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate runs even if cached results exist")
    parser.add_argument("--runs", type=int, default=NUM_MONTE_CARLO_RUNS, help="Monte Carlo runs per cell")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "oat_sensitivity", help="Output directory")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS, help="Parallel jobs (-1 = all cores)")
    args = parser.parse_args()

    partners = list(MANUSCRIPT_PARTNERS)
    print("OAT design (parameter: baseline, low, high):")
    for param, (kw, b, lo, hi) in OAT_DESIGN.items():
        print(f"  {param:8s} {kw:16s} {fmt_value(b):>8s} {fmt_value(lo):>8s} {fmt_value(hi):>8s}")
    print(f"Benchmarks at theta in {OAT_THETA_BENCHMARK_GRID}. Partners: {len(partners)}, runs: {args.runs}")

    per_run = run_sweep(partners, args.runs, args.out, args.overwrite, args.n_jobs)
    check_against_cached(per_run, partners, args.runs)
    eff = effects(per_run, partners)
    theta_tab = theta_benchmarks(per_run, partners)
    print_tables(eff, theta_tab, partners)
    order = ordering_robustness(per_run, partners)
    print_ordering(order, partners)

    eff.round(5).to_csv(args.out / "effects.csv", index=False)
    theta_tab.round(5).to_csv(args.out / "theta_benchmarks.csv", index=False)
    order.round(5).to_csv(args.out / "ordering_robustness.csv", index=False)
    print(f"\nSaved effects.csv, theta_benchmarks.csv and ordering_robustness.csv to {args.out}")


if __name__ == "__main__":
    main()
