#!/usr/bin/env python3
"""
Monte Carlo convergence check for the number of runs N = 300.

From the stored main runs of run_benchmark_comparison.py (no new simulation): for the trust-minus-benchmark payoff
difference on every partner, against the cumulative, discounted (rho = 0.9) and window (w = 10)
benchmarks, it reports how the estimate and its conclusion behave as N grows.

For each N in N_GRID and each of the 27 contrasts:
    halfwidth   95% Welch CI half-width of the difference using N runs per agent
    agree       share of random subsets of size N whose significance sign (+, -, n.s.) equals the one
                at N = 300 (random subsets, so the result does not depend on run order)
    drift       mean |estimate at N - estimate at 300| over the random subsets

and, per contrast, the N needed for the 95% CI half-width to fall below 1 payoff point, extrapolated
from the run-to-run standard deviations (halfwidth ~ 1/sqrt(N)).

Outputs: results/convergence/convergence_by_n.csv, convergence_contrasts.csv

Usage:
    python scripts/check_mc_convergence.py
"""

import numpy as np
import pandas as pd

from trust_bayesian_agent_comparison.analysis import per_run_metrics, sig_mark
from trust_bayesian_agent_comparison.analysis.batch import Z95
from trust_bayesian_agent_comparison.partners import MANUSCRIPT_PARTNERS
from trust_bayesian_agent_comparison.config import (
    RESULTS_DIR,
    NUM_MONTE_CARLO_RUNS,
    RECENCY_BENCHMARK_KEY,
    SLIDING_WINDOW_W,
)

B1_DIR = RESULTS_DIR / "benchmarks"
OUT_DIR = RESULTS_DIR / "convergence"
BENCHMARKS = ("cumulative", RECENCY_BENCHMARK_KEY, f"window_w{SLIDING_WINDOW_W}")
N_GRID = (25, 50, 100, 150, 200, 250, 300)
N_SUBSETS = 500
TARGET_HALFWIDTH = 1.0


def diff_ci(t, b):
    d = t.mean() - b.mean()
    hw = Z95 * np.sqrt(t.var(ddof=1) / len(t) + b.var(ddof=1) / len(b))
    return d, d - hw, d + hw, hw


def main():
    rng = np.random.default_rng(0)
    trust = {p: per_run_metrics(pd.read_csv(B1_DIR / "cumulative" / f"{p}_agent1.csv")).Pi.to_numpy()
             for p in MANUSCRIPT_PARTNERS}
    rows, contrasts = [], []
    for p in MANUSCRIPT_PARTNERS:
        for b in BENCHMARKS:
            bench = per_run_metrics(pd.read_csv(B1_DIR / b / f"{p}_agent2.csv")).Pi.to_numpy()
            t = trust[p]
            assert len(t) == len(bench) == NUM_MONTE_CARLO_RUNS
            d300, lo300, hi300, hw300 = diff_ci(t, bench)
            sig300 = sig_mark(lo300, hi300)
            sd_t, sd_b = t.std(ddof=1), bench.std(ddof=1)
            n_target = int(np.ceil((Z95 / TARGET_HALFWIDTH) ** 2 * (sd_t ** 2 + sd_b ** 2)))
            contrasts.append({"partner": p, "benchmark": b, "d300": d300, "halfwidth300": hw300, "sig300": sig300,
                              "sd_trust": sd_t, "sd_bench": sd_b, "n_for_halfwidth_1": n_target})
            for n in N_GRID:
                agree, drift, hws = [], [], []
                for _ in range(N_SUBSETS if n < NUM_MONTE_CARLO_RUNS else 1):
                    it = rng.choice(len(t), n, replace=False) if n < len(t) else np.arange(len(t))
                    ib = rng.choice(len(bench), n, replace=False) if n < len(bench) else np.arange(len(bench))
                    d, lo, hi, hw = diff_ci(t[it], bench[ib])
                    agree.append(sig_mark(lo, hi) == sig300)
                    drift.append(abs(d - d300))
                    hws.append(hw)
                rows.append({"partner": p, "benchmark": b, "N": n, "agree": np.mean(agree),
                             "drift": np.mean(drift), "halfwidth": np.mean(hws)})
    by_n = pd.DataFrame(rows)
    con = pd.DataFrame(contrasts)

    print("Trust-minus-benchmark total payoff, 27 contrasts (9 partners x 3 benchmarks), "
          f"{N_SUBSETS} random subsets per N\n")
    summary = by_n.groupby("N").agg(agree_mean=("agree", "mean"), agree_min=("agree", "min"),
                                    halfwidth_median=("halfwidth", "median"), halfwidth_max=("halfwidth", "max"),
                                    drift_median=("drift", "median"))
    print(f"{'N':>5s}{'conclusion = N=300 (mean)':>28s}{'(worst contrast)':>18s}"
          f"{'CI half-width median':>22s}{'max':>8s}{'|est - est300| median':>24s}")
    for n, r in summary.iterrows():
        print(f"{n:5d}{100 * r.agree_mean:27.1f}%{100 * r.agree_min:17.1f}%{r.halfwidth_median:22.2f}"
              f"{r.halfwidth_max:8.2f}{r.drift_median:24.2f}")

    print(f"\nPer contrast at N = 300: difference, 95% CI half-width, and the N needed for a half-width of "
          f"{TARGET_HALFWIDTH:g} payoff point")
    print(f"{'Partner':22s}" + "".join(f"{b[:16]:>30s}" for b in BENCHMARKS))
    for p in MANUSCRIPT_PARTNERS:
        cells = []
        for b in BENCHMARKS:
            r = con[(con.partner == p) & (con.benchmark == b)].iloc[0]
            cells.append(f"{r.d300:+6.1f} +/-{r.halfwidth300:4.1f}{r.sig300 or ' '}  N1={r.n_for_halfwidth_1:6d}")
        print(f"{p:22s}" + "".join(f"{c:>30s}" for c in cells))

    unstable = by_n[(by_n.N == 150) & (by_n.agree < 0.95)]
    print(f"\nContrasts whose conclusion at N = 150 matches N = 300 in fewer than 95% of subsets: {len(unstable)}")
    if len(unstable):
        print(unstable.merge(con, on=["partner", "benchmark"])[["partner", "benchmark", "agree", "d300",
                                                                  "halfwidth300", "sig300"]].round(2).to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_n.round(4).to_csv(OUT_DIR / "convergence_by_n.csv", index=False)
    con.round(4).to_csv(OUT_DIR / "convergence_contrasts.csv", index=False)
    print(f"\nSaved convergence_by_n.csv and convergence_contrasts.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
