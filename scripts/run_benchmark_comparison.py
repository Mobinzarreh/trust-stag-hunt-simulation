#!/usr/bin/env python3
"""
Compare the trust-sensitive agent against cumulative and adaptive Beta benchmarks

All benchmarks share the Beta(1,1) start, posterior-mean estimate, payoff matrix and logit
action rule of the submitted Bayesian benchmark. Only the memory differs:
    cumulative           BayesianFocalAgent, all outcomes weighted equally (submitted benchmark)
    discounted_rho<r>    DiscountedBetaAgent, exponential forgetting, rho in BENCHMARK_RHO_GRID
    disc_pp_rho<r>       PriorPreservingDiscountedBetaAgent, forgets the evidence but keeps the prior fixed
                         (the discounted Bayesian agent of the paper, see config.RECENCY_BENCHMARK_KEY)
    window_w<w>          SlidingWindowBetaAgent, last w outcomes, w in BENCHMARK_WINDOW_GRID
The primary, memory-matched pair is rho = gamma_x = 0.9 (prior-preserving) and w = 10 ~ 1/(1 - rho).
The other settings, including the plain discounted model that decays its prior, are a sensitivity
analysis. Nothing is tuned per partner.

Each benchmark runs as agent2 against FocalAgent as agent1, with the same base seed, and writes
to <out>/<benchmark>/<partner>_agent{1,2}.csv. Agent1 always gets the same RNG stream, so the
trust agent's runs are identical across pairings (checked below), and every benchmark sees the
same agent2 stream (common random numbers across benchmarks).

Partners use the revised configuration (Gradual Deterioration over the full 100 rounds); the originally
submitted configuration is available through submitted_v1=True.

Usage:
    python scripts/run_benchmark_comparison.py                  # reuse cached runs where present
    python scripts/run_benchmark_comparison.py --overwrite      # regenerate every run
    python scripts/run_benchmark_comparison.py --runs 20 --out results/benchmarks_smoke --overwrite
"""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from trust_bayesian_agent_comparison.agents import (
    FocalAgent,
    BayesianFocalAgent,
    DiscountedBetaAgent,
    SlidingWindowBetaAgent,
    PriorPreservingDiscountedBetaAgent,
    ChangePointBetaAgent,
    ThompsonCumulativeBetaAgent,
    ThompsonWindowBetaAgent,
)
from trust_bayesian_agent_comparison.analysis import MonteCarloManager, per_run_metrics
from trust_bayesian_agent_comparison.partners import MANUSCRIPT_PARTNERS, PARTNER_CATEGORY, partner_factory
from trust_bayesian_agent_comparison.simulation import run_agent_simulation
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
    BENCHMARK_RHO_GRID,
    BENCHMARK_WINDOW_GRID,
    CHANGEPOINT_HAZARD,
    BENCHMARK_HAZARD_GRID,
)

TRUST = "trust"

# Benchmark key -> (display label, picklable agent factory)
BENCHMARKS = {"cumulative": ("Cumulative", BayesianFocalAgent)}
BENCHMARKS.update({f"discounted_rho{r:.2f}": (f"Disc rho={r:.2f}", partial(DiscountedBetaAgent, rho=r))
                   for r in BENCHMARK_RHO_GRID})
BENCHMARKS.update({f"window_w{w}": (f"Window w={w}", partial(SlidingWindowBetaAgent, window=w))
                   for w in BENCHMARK_WINDOW_GRID})

BENCHMARKS[RECENCY_BENCHMARK_KEY] = (RECENCY_BENCHMARK_LABEL,
                                     partial(PriorPreservingDiscountedBetaAgent, rho=DISCOUNTED_BETA_RHO))

PRIMARY = ("cumulative", RECENCY_BENCHMARK_KEY, f"window_w{SLIDING_WINDOW_W}")
LABELS = {TRUST: "Trust", **{k: v[0] for k, v in BENCHMARKS.items()}}

LIMIT_CASES = [partial(DiscountedBetaAgent, rho=1.0), partial(SlidingWindowBetaAgent, window=NUM_ROUNDS),
               partial(PriorPreservingDiscountedBetaAgent, rho=1.0)]
LIMIT_DESCRIPTION = ["rho=1", "w=K", "prior-preserving rho=1"]

EXTRA_BENCHMARKS = {
    **{f"changepoint_h{h:.2f}": (f"CP h={h:.2f}", partial(ChangePointBetaAgent, hazard=h))
       for h in BENCHMARK_HAZARD_GRID},
    "ts_cumulative": ("TS cumulative", ThompsonCumulativeBetaAgent),
    f"ts_window_w{SLIDING_WINDOW_W}": (f"TS window w={SLIDING_WINDOW_W}",
                                       partial(ThompsonWindowBetaAgent, window=SLIDING_WINDOW_W)),
}
# Memory-matched extras (h = 1 - rho) plus both Thompson variants join the per-metric tables
EXTRA_PRIMARY = (f"changepoint_h{CHANGEPOINT_HAZARD:.2f}", "ts_cumulative", f"ts_window_w{SLIDING_WINDOW_W}")


def enable_extras():
    """Add the extension benchmarks to the run, the tables and the limiting-case checks."""
    global PRIMARY
    BENCHMARKS.update(EXTRA_BENCHMARKS)
    LABELS.update({k: v[0] for k, v in EXTRA_BENCHMARKS.items()})
    PRIMARY = PRIMARY + EXTRA_PRIMARY
    LIMIT_CASES.extend([partial(ChangePointBetaAgent, hazard=0.0)])
    LIMIT_DESCRIPTION.extend(["h=0"])


# Columns that must match exactly when two runs are claimed to be identical
TRAJECTORY_COLS = ["round", "agent_belief", "agent_action", "partner_action", "agent_payoff", "run_id"]

# Partners whose submitted configuration equals the corrected one
UNCHANGED_SINCE_SUBMISSION = tuple(p for p in MANUSCRIPT_PARTNERS
                                   if p not in ("GradualDeterioration", "StrategicCheater"))


def check_limiting_cases(partners, seeds=range(5)):
    """Each benchmark's no-forgetting limit must reproduce the cumulative benchmark bit for bit."""
    for p in partners:
        for i in seeds:
            seed = MC_BASE_SEED + i
            ref = run_agent_simulation(BayesianFocalAgent(), partner_factory(p)(), NUM_ROUNDS, seed)
            for factory in LIMIT_CASES:
                agent = factory()
                df = run_agent_simulation(agent, partner_factory(p)(), NUM_ROUNDS, seed)
                if not ref[TRAJECTORY_COLS[:-1]].equals(df[TRAJECTORY_COLS[:-1]]):
                    raise AssertionError(f"{agent.name} limit differs from cumulative on {p}, seed {seed}")
    print(f"Limiting cases: {', '.join(LIMIT_DESCRIPTION)} reproduce the cumulative benchmark exactly.")


def run_all(partners, num_runs, out_dir, overwrite, n_jobs):
    """Run every benchmark pairing. Returns {key: {partner: DataFrame}}, trust runs under TRUST."""
    trust_runs = {key: {} for key in BENCHMARKS}
    bench_runs = {key: {} for key in BENCHMARKS}
    for key, (label, factory) in BENCHMARKS.items():
        mc = MonteCarloManager(results_dir=out_dir / key)
        for p in partners:
            print(f"\n--- {label} vs {p}")
            mc.run_monte_carlo(
                agent1_factory=FocalAgent,
                agent2_factory=factory,
                partner_factory=partner_factory(p, NUM_ROUNDS),
                partner_name=p,
                num_runs=num_runs,
                base_seed=MC_BASE_SEED,
                num_rounds=NUM_ROUNDS,
                n_jobs=n_jobs,
                overwrite=overwrite,
            )
            # Always analyse what is on disk. pandas' default CSV float parser can be 1 ulp off the
            # in-memory value, so mixing fresh and cached frames would break the identity checks.
            df1, df2 = (pd.read_csv(mc.results_dir / f"{p}_{a}.csv") for a in ("agent1", "agent2"))
            # MonteCarloManager loads cached CSVs without checking how they were made
            for df in (df1, df2):
                if df["run_id"].nunique() != num_runs or df["meta_base_seed"].iloc[0] != MC_BASE_SEED:
                    raise RuntimeError(f"Cached results in {out_dir / key} do not match --runs {num_runs}; "
                                       f"rerun with --overwrite")
            trust_runs[key][p], bench_runs[key][p] = df1, df2
    return trust_runs, bench_runs


def check_trust_identical(trust_runs, partners):
    """The trust agent must produce the same runs in every pairing."""
    first, *rest = list(BENCHMARKS)
    for p in partners:
        ref = trust_runs[first][p][TRAJECTORY_COLS].to_numpy(float)
        for key in rest:
            if not np.array_equal(ref, trust_runs[key][p][TRAJECTORY_COLS].to_numpy(float)):
                raise AssertionError(f"Trust agent runs differ between '{first}' and '{key}' pairings on {p}")
    print(f"Trust agent runs identical across all {len(BENCHMARKS)} pairings.")


def check_against_submitted(trust_runs, bench_runs, partners, num_runs):
    """For partners unchanged since submission, the cumulative pairing must equal the cached main study."""
    if num_runs != NUM_MONTE_CARLO_RUNS:
        return
    checked = []
    for p in (p for p in partners if p in UNCHANGED_SINCE_SUBMISSION):
        files = (RESULTS_DIR / f"{p}_agent1.csv", RESULTS_DIR / f"{p}_agent2.csv")
        if not all(f.exists() for f in files):
            continue
        for df, f in zip((bench_runs["cumulative"][p],), (files[1],)):
            cached = pd.read_csv(f)[TRAJECTORY_COLS].to_numpy(float)
            if not np.array_equal(df[TRAJECTORY_COLS].to_numpy(float), cached):
                raise AssertionError(f"Cumulative pairing on {p} differs from the submitted results in {f}")
        checked.append(p)
    if checked:
        print(f"Cumulative benchmark runs identical to the submitted main study on {len(checked)} partners "
              f"(trust side not compared: the revised agent differs by design, see submitted_v1).")


def summarize(trust_runs, bench_runs, partners):
    """Long table: one row per partner x model, with differences trust - model and 95% CIs."""
    rows = []
    for p in partners:
        ts = per_run_metrics(trust_runs["cumulative"][p])
        models = {TRUST: ts, **{key: per_run_metrics(bench_runs[key][p]) for key in BENCHMARKS}}
        for key, m in models.items():
            row = {
                "partner": p, "category": PARTNER_CATEGORY[p], "model": key, "label": LABELS[key],
                "Pi": m.Pi.mean(), "Pi_sd": m.Pi.std(), "HB": m.HB.mean(),
                "EF": 100 * m.Pi.mean() / m.HB.mean(),
                "SC": 100 * m.SC.mean(), "SC_sd": 100 * m.SC.std(),
                "FC": 100 * m.FC.mean(), "FC_sd": 100 * m.FC.std(),
            }
            # Trust and benchmark runs use independent RNG streams, so the difference of means
            # gets an unpaired (Welch) standard error
            for metric, scale in (("Pi", 1), ("SC", 100), ("FC", 100)):
                diff = scale * (ts[metric].mean() - m[metric].mean())
                se = scale * np.sqrt(ts[metric].var() / len(ts) + m[metric].var() / len(m))
                row[f"d{metric}"], row[f"d{metric}_lo"], row[f"d{metric}_hi"] = diff, diff - 1.96 * se, diff + 1.96 * se
            rows.append(row)
    return pd.DataFrame(rows)


def sig_mark(row, metric):
    """'+' if trust is significantly higher, '-' if significantly lower, '' otherwise."""
    if row[f"d{metric}_lo"] > 0:
        return "+"
    if row[f"d{metric}_hi"] < 0:
        return "-"
    return ""


def print_tables(summary, partners):
    keys = [TRUST, *BENCHMARKS]
    by = summary.set_index(["partner", "model"])

    def wide(metric, models, fmt="{:7.1f}", with_sig=False):
        header = f"{'Partner':22s} {'category':15s} " + " ".join(f"{LABELS[k]:>16s}" for k in models)
        print(header)
        for p in partners:
            cells = []
            for k in models:
                r = by.loc[(p, k)]
                mark = sig_mark(r, metric) if (with_sig and k != TRUST) else ""
                cells.append(f"{fmt.format(r[metric]) + mark:>16s}")
            print(f"{p:22s} {PARTNER_CATEGORY[p]:15s} " + " ".join(cells))

    for metric, title in (("Pi", "Total payoff"), ("EF", "Efficiency EF (%)"),
                          ("SC", "Successful collaboration SC (%)"), ("FC", "Failed collaboration FC (%)")):
        print(f"\n{'=' * 110}\n{title}, primary memory-matched comparison"
              f"{'  (+/- : trust significantly higher/lower, 95% CI)' if metric != 'EF' else ''}\n{'=' * 110}")
        wide(metric, [TRUST, *PRIMARY], with_sig=metric != "EF")

    print(f"\n{'=' * 110}\nTotal payoff, all benchmark settings (+/- : trust significantly higher/lower)\n{'=' * 110}")
    wide("Pi", keys, with_sig=True)

    print(f"\n{'=' * 110}\nTrust minus benchmark, total payoff: mean [95% CI]\n{'=' * 110}")
    print(f"{'Partner':22s} " + " ".join(f"{LABELS[k]:>22s}" for k in BENCHMARKS))
    for p in partners:
        cells = [f"{by.loc[(p, k), 'dPi']:+6.1f} [{by.loc[(p, k), 'dPi_lo']:+6.1f},{by.loc[(p, k), 'dPi_hi']:+6.1f}]"
                 for k in BENCHMARKS]
        print(f"{p:22s} " + " ".join(f"{c:>22s}" for c in cells))

    print(f"\n{'=' * 110}\nPartners where trust is significantly higher / not different / lower (total payoff)\n{'=' * 110}")
    for k in BENCHMARKS:
        marks = [sig_mark(by.loc[(p, k)], "Pi") for p in partners]
        nonstat = [sig_mark(by.loc[(p, k)], "Pi") for p in partners if PARTNER_CATEGORY[p] == "non-stationary"]
        print(f"  {LABELS[k]:16s} all: {marks.count('+')}/{marks.count('')}/{marks.count('-')}   "
              f"non-stationary only: {nonstat.count('+')}/{nonstat.count('')}/{nonstat.count('-')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate runs even if cached CSVs exist")
    parser.add_argument("--runs", type=int, default=NUM_MONTE_CARLO_RUNS, help="Monte Carlo runs per partner")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "benchmarks", help="Output directory")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS, help="Parallel jobs (-1 = all cores)")
    parser.add_argument("--extras", action="store_true",
                        help="Also run the extension benchmarks (prior-preserving discounting, "
                             "Bayesian change-point detection, Thompson sampling)")
    args = parser.parse_args()

    if args.extras:
        enable_extras()

    partners = list(MANUSCRIPT_PARTNERS)
    print(f"Benchmarks: {', '.join(LABELS[k] for k in BENCHMARKS)}")
    print(f"Primary memory-matched pair: {LABELS[PRIMARY[1]]}, {LABELS[PRIMARY[2]]}")
    print(f"Partners: {len(partners)}, runs: {args.runs}, rounds: {NUM_ROUNDS}, base seed: {MC_BASE_SEED}")

    check_limiting_cases(partners)
    trust_runs, bench_runs = run_all(partners, args.runs, args.out, args.overwrite, args.n_jobs)
    check_trust_identical(trust_runs, partners)
    check_against_submitted(trust_runs, bench_runs, partners, args.runs)

    summary = summarize(trust_runs, bench_runs, partners)
    print_tables(summary, partners)

    out = args.out / ("benchmark_summary_extended.csv" if args.extras else "benchmark_summary.csv")
    summary.round(4).to_csv(out, index=False)
    print(f"\nSummary saved to: {out}")


if __name__ == "__main__":
    main()
