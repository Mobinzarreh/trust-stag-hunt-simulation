#!/usr/bin/env python3
"""
Sensitivity of the trust-sensitive agent to its confidence ceiling and its initialization.
Its runs are also the cross-check for the t_max, x_0 and t_0 rows of run_oat_sensitivity.py.

Two sweeps, each variant run with 300 Monte Carlo runs on all manuscript partners:
    ceiling   t_max in T_MAX_GRID (eps = 1). The estimate is bounded by (eps + t_max)/(2 eps + t_max),
              so this tests whether the ceiling, rather than shrinkage toward 0.5 as such, produces the
              collaboration gap against the Bayesian benchmark.
    init      x_0 in INIT_X0_GRID by t_0 in INIT_T0_GRID. Tests when and how the initial belief and
              confidence influence decisions, and whether the findings depend on them.
              Agent2 is the cumulative Bayesian benchmark with the matched prior
              Beta(eps + x_0 t_0, eps + (1 - x_0) t_0), which has the same initial estimate and
              concentration, so each initialization is compared on equal terms.

p-hat = (eps + x t)/(2 eps + t) and t scales linearly with t_max, so p-hat depends on t_max and eps only
through t_max / eps. One extra variant (eps = 0.5, t_max = 10) checks this against t_max = 20.

Every trust variant runs as agent1, the RNG stream of the full model in the main study and the ablation, so
differences from the full model are paired. Partner randomness uses Python's `random` module, separate
from the agent's numpy draws, so non-reactive partners produce the same actions for every variant and
the effect of initialization on p-hat can be traced exactly.

Usage:
    python scripts/run_bounds_init_sensitivity.py                # reuse cached runs where present
    python scripts/run_bounds_init_sensitivity.py --overwrite    # regenerate every run
    python scripts/run_bounds_init_sensitivity.py --runs 20 --out results/bounds_init_smoke --overwrite
"""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from trust_bayesian_agent_comparison.agents import FocalAgent, BayesianFocalAgent
from trust_bayesian_agent_comparison.analysis import MonteCarloManager, per_run_metrics
from trust_bayesian_agent_comparison.partners import MANUSCRIPT_PARTNERS, PARTNER_CATEGORY, partner_factory
from trust_bayesian_agent_comparison.config import (
    RESULTS_DIR,
    NUM_MONTE_CARLO_RUNS,
    NUM_ROUNDS,
    MC_BASE_SEED,
    N_JOBS,
    EPS,
    TRUST_MAX,
    AGENT_T_MAX,
    PAYOFF_MATRIX,
    INVERSE_TEMPERATURE,
    DECISION_THRESHOLD,
    RECENCY_BENCHMARK_KEY,
    RECENCY_BENCHMARK_LABEL,
    SLIDING_WINDOW_W,
    T_MAX_GRID,
    INIT_X0_GRID,
    INIT_T0_GRID,
)

X0_DEFAULT = 1 - DECISION_THRESHOLD   # FocalAgent's default initial belief
FULL_DIR = RESULTS_DIR / "ablation" / "full"
BENCH_DIR = RESULTS_DIR / "benchmarks"
REFERENCES = {
    "cumulative": "Cumulative",
    RECENCY_BENCHMARK_KEY: RECENCY_BENCHMARK_LABEL,
    f"window_w{SLIDING_WINDOW_W}": f"Window w={SLIDING_WINDOW_W}",
}
# Partners whose actions do not depend on the agent's behavior
EXOGENOUS = ("AlwaysCooperate", "AlwaysDefect", "Random", "SingleCycle", "GradualDeterioration")
DECAY_ROUNDS = (0, 1, 2, 3, 5, 10, 20)
TRAJECTORY_COLS = ["round", "agent_belief", "agent_action", "partner_action", "agent_payoff", "run_id"]


def init_key(x0, t0):
    return f"x{x0:.3f}_t{t0:g}"


def ceiling_variants():
    variants = {f"tmax{v:g}": (f"t_max={v:g}", dict(t_max=v), BayesianFocalAgent) for v in T_MAX_GRID}
    variants["eps0.5"] = ("eps=0.5", dict(eps=0.5), BayesianFocalAgent)
    return variants


def init_variants():
    return {init_key(x0, t0): (f"x0={x0:.2f} t0={t0:g}", dict(u_i=x0, t_init=t0),
                               partial(BayesianFocalAgent, alpha0=EPS + x0 * t0, beta0=EPS + (1 - x0) * t0))
            for x0 in INIT_X0_GRID for t0 in INIT_T0_GRID}


def propensity(p):
    """P(collaborate) under the shared logit rule, from the estimate p."""
    ev_i = (1 - p) * PAYOFF_MATRIX[0, 0, 0] + p * PAYOFF_MATRIX[0, 1, 0]
    ev_c = (1 - p) * PAYOFF_MATRIX[1, 0, 0] + p * PAYOFF_MATRIX[1, 1, 0]
    return 1.0 / (1.0 + np.exp(-INVERSE_TEMPERATURE * (ev_c - ev_i)))


def run_sweep(variants, partners, num_runs, out_dir, overwrite, n_jobs):
    """Run every variant. Returns {key: {partner: (trust df, matched-benchmark df)}}."""
    runs = {}
    for key, (label, overrides, bench_factory) in variants.items():
        mc = MonteCarloManager(results_dir=out_dir / key)
        runs[key] = {}
        for p in partners:
            print(f"\n--- {label} vs {p}")
            mc.run_monte_carlo(
                agent1_factory=partial(FocalAgent, **overrides),
                agent2_factory=bench_factory,
                partner_factory=partner_factory(p, NUM_ROUNDS),
                partner_name=p,
                num_runs=num_runs,
                base_seed=MC_BASE_SEED,
                num_rounds=NUM_ROUNDS,
                n_jobs=n_jobs,
                overwrite=overwrite,
            )
            # Analyse what is on disk so fresh and cached runs are parsed the same way
            dfs = tuple(pd.read_csv(mc.results_dir / f"{p}_{a}.csv") for a in ("agent1", "agent2"))
            for df in dfs:
                if df["run_id"].nunique() != num_runs or df["meta_base_seed"].iloc[0] != MC_BASE_SEED:
                    raise RuntimeError(f"Cached results in {out_dir / key} do not match --runs {num_runs}; "
                                       f"rerun with --overwrite")
            runs[key][p] = dfs
    return runs


def same(a, b, cols=TRAJECTORY_COLS):
    return np.array_equal(a[cols].to_numpy(float), b[cols].to_numpy(float))


def run_checks(ceiling, init, partners, num_runs):
    default_init = init_key(X0_DEFAULT, 0.0)
    for p in partners:
        if not same(ceiling[f"tmax{AGENT_T_MAX:g}"][p][0], init[default_init][p][0]):
            raise AssertionError(f"Default settings differ between the two sweeps on {p}")
        twice = 2 * AGENT_T_MAX
        if not same(ceiling[f"tmax{twice:g}"][p][0], ceiling["eps0.5"][p][0]):
            raise AssertionError(f"eps = 0.5 does not reproduce t_max = {twice:g} on {p}")
    print(f"Default t_max and default initialization give identical runs; eps=0.5 reproduces t_max={2 * AGENT_T_MAX:g} exactly "
          "(p-hat depends only on t_max/eps).")
    if num_runs == NUM_MONTE_CARLO_RUNS:
        matched = sum(same(ceiling[f"tmax{AGENT_T_MAX:g}"][p][0], pd.read_csv(FULL_DIR / f"{p}_agent1.csv"))
                      for p in partners if (FULL_DIR / f"{p}_agent1.csv").exists())
        print(f"Default settings identical to the full-model runs of B1/B2 on {matched} partners.")


def paired_diff(a, b):
    """Mean of a - b over runs with a 95% CI (a, b indexed by run_id)."""
    d = a - b
    se = d.std() / np.sqrt(len(d))
    return d.mean(), d.mean() - 1.96 * se, d.mean() + 1.96 * se


def welch(a, b):
    d = a.mean() - b.mean()
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    return d, d - 1.96 * se, d + 1.96 * se


def mark(lo, hi):
    return "*" if lo > 0 or hi < 0 else ""


def summarize(sweep_name, runs, base_key, partners):
    rows = []
    for key, by_partner in runs.items():
        for p in partners:
            m = per_run_metrics(by_partner[p][0])
            base = per_run_metrics(runs[base_key][p][0])
            bench = per_run_metrics(by_partner[p][1])
            row = {"sweep": sweep_name, "variant": key, "partner": p, "category": PARTNER_CATEGORY[p],
                   "Pi": m.Pi.mean(), "SC": 100 * m.SC.mean(), "FC": 100 * m.FC.mean(),
                   "bench_Pi": bench.Pi.mean()}
            for metric, scale in (("Pi", 1), ("SC", 100), ("FC", 100)):
                d, lo, hi = paired_diff(scale * m[metric], scale * base[metric])
                row.update({f"d{metric}": d, f"d{metric}_lo": lo, f"d{metric}_hi": hi})
            d, lo, hi = welch(m.Pi, bench.Pi)
            row.update(vs_bench=d, vs_bench_lo=lo, vs_bench_hi=hi)
            rows.append(row)
    return pd.DataFrame(rows)


def print_ceiling(summary, refs, partners):
    s = summary[summary.sweep == "ceiling"].set_index(["variant", "partner"])
    keys = [f"tmax{v:g}" for v in T_MAX_GRID]
    print(f"\n{'=' * 120}\nCEILING SWEEP: t_max (eps = 1)\n{'=' * 120}")
    print(f"{'':22s}" + "".join(f"{'t_max=' + format(v, 'g'):>13s}" for v in T_MAX_GRID)
          + "".join(f"{lab:>15s}" for k, lab in REFERENCES.items() if k in refs))
    ceil = [(EPS + v) / (2 * EPS + v) for v in T_MAX_GRID]
    print(f"{'max p-hat':22s}" + "".join(f"{c:13.3f}" for c in ceil))
    print(f"{'max P(collaborate)':22s}" + "".join(f"{propensity(c):13.3f}" for c in ceil))
    for metric, title in (("SC", "SC (%)"), ("Pi", "Total payoff"), ("FC", "FC (%)")):
        print(f"\n{title}  (* = differs from t_max = {AGENT_T_MAX:g}, paired 95% CI)")
        for p in partners:
            cells = []
            for k in keys:
                r = s.loc[(k, p)]
                star = mark(r[f"d{metric}_lo"], r[f"d{metric}_hi"]) if k != f"tmax{AGENT_T_MAX:g}" else ""
                cells.append(f"{r[metric]:12.1f}{star:1s}")
            ref_cells = [f"{refs[k][p][metric]:15.1f}" for k in REFERENCES if k in refs]
            print(f"{p:22s}" + "".join(cells) + "".join(ref_cells))


def print_init(summary, init, partners):
    s = summary[summary.sweep == "init"].set_index(["variant", "partner"])
    keys = list(init_variants())
    base = init_key(X0_DEFAULT, 0.0)
    print(f"\n{'=' * 120}\nINITIALIZATION SWEEP: x_0 by t_0\n{'=' * 120}")
    print("Round-1 estimate p-hat_0 = (eps + x_0 t_0)/(2 eps + t_0) and P(collaborate)")
    print(f"{'':10s}" + "".join(f"{'t0=' + format(t, 'g'):>16s}" for t in INIT_T0_GRID))
    for x0 in INIT_X0_GRID:
        cells = []
        for t0 in INIT_T0_GRID:
            p0 = (EPS + x0 * t0) / (2 * EPS + t0)
            cells.append(f"{p0:.3f} / {propensity(p0):.3f}")
        print(f"x0={x0:.3f} " + "".join(f"{c:>16s}" for c in cells))

    print(f"\nTotal payoff, change vs the submitted initialization (x0={X0_DEFAULT:.3f}, t0=0), paired")
    print(f"{'variant':22s}" + "".join(f"{p[:13]:>14s}" for p in partners))
    for k in keys:
        if k == base:
            continue
        cells = [f"{s.loc[(k, p), 'dPi']:+.1f}{mark(s.loc[(k, p), 'dPi_lo'], s.loc[(k, p), 'dPi_hi'])}"
                 for p in partners]
        print(f"{k:22s}" + "".join(f"{c:>14s}" for c in cells))
    rng = s.reset_index().groupby("partner").dPi.agg(["min", "max"])
    print(f"\n{'range over all inits':22s}" + "".join(
        f"{format(rng.loc[p, 'min'], '+.1f') + '..' + format(rng.loc[p, 'max'], '+.1f'):>14s}" for p in partners))

    print("\nTrust minus matched-prior Bayesian, total payoff (unpaired; * = 95% CI excludes 0)")
    print(f"{'variant':22s}" + "".join(f"{p[:13]:>14s}" for p in partners))
    for k in keys:
        cells = [f"{s.loc[(k, p), 'vs_bench']:+.1f}{mark(s.loc[(k, p), 'vs_bench_lo'], s.loc[(k, p), 'vs_bench_hi'])}"
                 for p in partners]
        print(f"{k:22s}" + "".join(f"{c:>14s}" for c in cells))

    print("\nHow long the initialization matters: mean |p-hat - p-hat(submitted init)| by round, "
          "non-reactive partners pooled")
    print(f"{'variant':22s}" + "".join(f"{'r=' + str(r):>8s}" for r in DECAY_ROUNDS) + f"{'first r < 0.01':>16s}")
    for k in keys:
        if k == base:
            continue
        diffs = []
        for p in EXOGENOUS:
            a = init[k][p][0].set_index(["run_id", "round"]).agent_belief
            b = init[base][p][0].set_index(["run_id", "round"]).agent_belief
            diffs.append((a - b).abs().groupby(level="round").mean())
        curve = pd.concat(diffs, axis=1).mean(axis=1)
        below = curve.index[(curve < 0.01) & (curve.index > 0)]
        settle = below[0] if len(below) else np.nan
        print(f"{k:22s}" + "".join(f"{curve.loc[r]:8.3f}" for r in DECAY_ROUNDS) + f"{settle:16}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate runs even if cached CSVs exist")
    parser.add_argument("--runs", type=int, default=NUM_MONTE_CARLO_RUNS, help="Monte Carlo runs per partner")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "bounds_init", help="Output directory")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS, help="Parallel jobs (-1 = all cores)")
    args = parser.parse_args()

    partners = list(MANUSCRIPT_PARTNERS)
    print(f"Ceiling grid t_max: {T_MAX_GRID}; init grid x0: {tuple(round(x, 3) for x in INIT_X0_GRID)}, "
          f"t0: {INIT_T0_GRID}")
    print(f"Partners: {len(partners)}, runs: {args.runs}, rounds: {NUM_ROUNDS}, base seed: {MC_BASE_SEED}")

    ceiling = run_sweep(ceiling_variants(), partners, args.runs, args.out / "ceiling", args.overwrite, args.n_jobs)
    init = run_sweep(init_variants(), partners, args.runs, args.out / "init", args.overwrite, args.n_jobs)
    run_checks(ceiling, init, partners, args.runs)

    summary = pd.concat([summarize("ceiling", ceiling, f"tmax{AGENT_T_MAX:g}", partners),
                         summarize("init", init, init_key(X0_DEFAULT, 0.0), partners)], ignore_index=True)
    refs = {}
    if args.runs == NUM_MONTE_CARLO_RUNS:
        for k in REFERENCES:
            files = {p: BENCH_DIR / k / f"{p}_agent2.csv" for p in partners}
            if all(f.exists() for f in files.values()):
                refs[k] = {p: {"Pi": (m := per_run_metrics(pd.read_csv(f))).Pi.mean(),
                               "SC": 100 * m.SC.mean(), "FC": 100 * m.FC.mean()} for p, f in files.items()}
    print_ceiling(summary, refs, partners)
    print_init(summary, init, partners)

    out = args.out / "bounds_init_summary.csv"
    summary.round(4).to_csv(out, index=False)
    print(f"\nSummary saved to: {out}")


if __name__ == "__main__":
    main()
