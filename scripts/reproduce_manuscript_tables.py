#!/usr/bin/env python3
"""
Reproduce Tables 4 and 5 of the submitted manuscript.

Recomputes, per Monte Carlo run and then aggregated over runs:
    Pi  total payoff                       (Eq. 17)
    SC  successful collaboration rate      (Eq. 18)
    FC  failed collaboration rate          (Eq. 19)
    HB  hindsight benchmark                (Eq. 20)
    EF  efficiency ratio, Pi / HB x 100    (Eq. 22)
and compares every cell against the values printed in the manuscript.

Partners are built with the registry's submitted_v1 configuration, which reproduces
the 70-round Gradual Deterioration horizon used for the submitted results.

Usage:
    python scripts/reproduce_manuscript_tables.py --rerun    # simulate the submitted model into results/reproduction/
                                                             # and compare with the values printed in the submitted paper
    python scripts/reproduce_manuscript_tables.py            # same comparison from runs already in results/
"""

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from trust_bayesian_agent_comparison.agents import FocalAgent, BayesianFocalAgent
from trust_bayesian_agent_comparison.analysis import MonteCarloManager, per_run_metrics
from trust_bayesian_agent_comparison.partners import partner_factory
from trust_bayesian_agent_comparison.config import (
    PAYOFF_MATRIX,
    RESULTS_DIR,
    NUM_MONTE_CARLO_RUNS,
    NUM_ROUNDS,
    MC_BASE_SEED,
)

REPRO_DIR = RESULTS_DIR / "reproduction"

# Manuscript partner label -> results file prefix
PARTNER_NAMES = {
    "Always Collaborate": "AlwaysCooperate",
    "Always Independent": "AlwaysDefect",
    "Random-(p)": "Random",
    "TFT": "TitForTat",
    "STFT": "SuspiciousTFT",
    "Single-Cycle": "SingleCycle",
    "Gradual Deterioration": "GradualDeterioration",
    "Strategic Cheater": "StrategicCheater",
    "Expectation Violating": "ExpectationViolation",
}

# Values as printed in the submitted manuscript.
# Table 4: Pi_TS, sd, HB_TS, sd, EF_TS, Pi_B, sd, HB_B, sd, EF_B
TABLE4 = {
    "Always Collaborate":    (378.9, 4.0, 400.0, 0.0, 94.7, 384.7, 3.6, 400.0, 0.0, 96.2),
    "Always Independent":    (193.2, 3.4, 200.0, 0.0, 96.6, 194.6, 3.2, 200.0, 0.0, 97.3),
    "Random-(p)":            (236.1, 10.0, 300.0, 9.6, 78.7, 237.4, 9.9, 300.6, 10.1, 79.0),
    "TFT":                   (198.0, 5.5, 217.8, 11.5, 90.9, 203.2, 19.9, 229.0, 32.2, 88.7),
    "STFT":                  (196.3, 2.6, 208.8, 5.8, 94.0, 196.9, 3.1, 208.5, 9.4, 94.4),
    "Single-Cycle":          (237.2, 4.9, 260.0, 0.0, 91.2, 209.0, 7.5, 260.0, 0.0, 80.4),
    "Gradual Deterioration": (242.7, 10.2, 288.6, 7.5, 84.1, 224.2, 0.7, 288.7, 8.3, 77.7),
    "Strategic Cheater":     (236.8, 48.0, 269.8, 61.8, 87.8, 229.8, 44.2, 274.0, 57.4, 83.9),
    "Expectation Violating": (271.9, 11.6, 334.5, 8.4, 81.3, 260.8, 14.5, 328.1, 10.6, 79.5),
}
TABLE4_COLS = ["Pi_TS", "Pi_TS_sd", "HB_TS", "HB_TS_sd", "EF_TS",
               "Pi_B", "Pi_B_sd", "HB_B", "HB_B_sd", "EF_B"]

# Table 5 (percent): SC_TS, sd, SC_B, sd, dSC, FC_TS, sd, FC_B, sd, dFC
TABLE5 = {
    "Always Collaborate":    (78.9, 4.0, 84.7, 3.6, -5.8, 0.0, 0.0, 0.0, 0.0, 0.0),
    "Always Independent":    (0.0, 0.0, 0.0, 0.0, 0.0, 3.4, 1.7, 2.7, 1.6, 0.7),
    "Random-(p)":            (13.9, 4.5, 14.3, 6.3, -0.5, 13.9, 3.3, 13.6, 4.0, 0.2),
    "TFT":                   (1.7, 3.2, 5.3, 11.8, -3.6, 6.3, 3.3, 8.3, 5.8, -2.0),
    "STFT":                  (0.2, 0.9, 0.4, 2.0, -0.2, 4.2, 2.5, 3.9, 3.1, 0.3),
    "Single-Cycle":          (22.4, 2.3, 23.7, 2.2, -1.3, 7.6, 2.2, 22.4, 3.7, -14.8),
    "Gradual Deterioration": (22.8, 4.8, 28.5, 5.8, -5.8, 12.2, 3.0, 24.4, 3.6, -12.2),
    "Strategic Cheater":     (19.6, 19.9, 20.7, 18.7, -1.2, 8.8, 3.5, 13.9, 6.1, -5.1),
    "Expectation Violating": (31.6, 2.8, 29.2, 3.5, 2.4, 13.4, 2.8, 16.2, 3.6, -2.8),
}
TABLE5_COLS = ["SC_TS", "SC_TS_sd", "SC_B", "SC_B_sd", "dSC",
               "FC_TS", "FC_TS_sd", "FC_B", "FC_B_sd", "dFC"]

# Manuscript values are printed to one decimal place
TOL = 0.05 + 1e-9



def compute_tables(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute Tables 4 and 5 from <partner>_agent{1,2}.csv in results_dir."""
    t4, t5, ef_alt = {}, {}, {}
    for label, name in PARTNER_NAMES.items():
        ts = per_run_metrics(pd.read_csv(results_dir / f"{name}_agent1.csv"))
        by = per_run_metrics(pd.read_csv(results_dir / f"{name}_agent2.csv"))
        t4[label] = (
            ts.Pi.mean(), ts.Pi.std(), ts.HB.mean(), ts.HB.std(), 100 * ts.Pi.mean() / ts.HB.mean(),
            by.Pi.mean(), by.Pi.std(), by.HB.mean(), by.HB.std(), 100 * by.Pi.mean() / by.HB.mean(),
        )
        t5[label] = tuple(100 * v for v in (
            ts.SC.mean(), ts.SC.std(), by.SC.mean(), by.SC.std(), ts.SC.mean() - by.SC.mean(),
            ts.FC.mean(), ts.FC.std(), by.FC.mean(), by.FC.std(), ts.FC.mean() - by.FC.mean(),
        ))
        # Alternative EF definition (mean of per-run ratios), to confirm which one the paper used
        ef_alt[label] = (100 * (ts.Pi / ts.HB).mean(), 100 * (by.Pi / by.HB).mean())
    return (pd.DataFrame.from_dict(t4, orient="index", columns=TABLE4_COLS),
            pd.DataFrame.from_dict(t5, orient="index", columns=TABLE5_COLS),
            pd.DataFrame.from_dict(ef_alt, orient="index", columns=["EF_TS_perrun", "EF_B_perrun"]))


def compare(computed: pd.DataFrame, printed: dict, cols: list[str], title: str) -> list[tuple]:
    """Print computed vs printed, cell by cell. Return list of mismatches."""
    ref = pd.DataFrame.from_dict(printed, orient="index", columns=cols)
    mismatches = []
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    print(f"{'Partner':22s} " + " ".join(f"{c:>9s}" for c in cols))
    for label in ref.index:
        print(f"{label:22s} " + " ".join(f"{computed.loc[label, c]:9.2f}" for c in cols))
        marks = []
        for c in cols:
            diff = computed.loc[label, c] - ref.loc[label, c]
            ok = abs(diff) <= TOL
            marks.append(f"{'ok' if ok else ref.loc[label, c]:>9}")
            if not ok:
                mismatches.append((title, label, c, ref.loc[label, c], computed.loc[label, c], diff))
        print(f"{'  printed':22s} " + " ".join(marks))
    return mismatches


def rerun_and_check_identity():
    """Regenerate the manuscript runs into REPRO_DIR and check they equal the cached CSVs."""
    mc = MonteCarloManager(results_dir=REPRO_DIR)
    numeric = ["round", "agent_belief", "agent_action", "partner_action", "agent_payoff", "run_id"]
    print(f"\n{'=' * 100}\nRE-RUN CHECK: fresh simulation vs cached CSVs\n{'=' * 100}")
    all_identical = True
    for name in PARTNER_NAMES.values():
        mc.run_monte_carlo(
            agent1_factory=partial(FocalAgent, submitted_v1=True),
            agent2_factory=BayesianFocalAgent,
            partner_factory=partner_factory(name, NUM_ROUNDS, submitted_v1=True),
            partner_name=name,
            num_runs=NUM_MONTE_CARLO_RUNS,
            base_seed=MC_BASE_SEED,
            num_rounds=NUM_ROUNDS,
            overwrite=True,
            notebook_compatible_seeding=False,
        )
        for suffix in ("agent1", "agent2"):
            if not (RESULTS_DIR / f"{name}_{suffix}.csv").exists():
                print(f"  {name:22s} {suffix}: no cached CSV in results/, identity check skipped")
                continue
            fresh = pd.read_csv(REPRO_DIR / f"{name}_{suffix}.csv")[numeric].to_numpy(float)
            cached = pd.read_csv(RESULTS_DIR / f"{name}_{suffix}.csv")[numeric].to_numpy(float)
            same_shape = fresh.shape == cached.shape
            max_diff = np.abs(fresh - cached).max() if same_shape else np.inf
            all_identical &= same_shape and max_diff == 0.0
            print(f"  {name:22s} {suffix}: shape match={same_shape}, max abs diff={max_diff:.3g}")
    print(f"\nFresh run identical to cached results: {all_identical}")
    return all_identical


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rerun", action="store_true",
                        help="Regenerate simulations into results/reproduction/ before comparing")
    args = parser.parse_args()

    source = RESULTS_DIR
    if args.rerun:
        rerun_and_check_identity()
        source = REPRO_DIR

    print(f"\nComputing tables from: {source}")
    t4, t5, ef_alt = compute_tables(source)

    mismatches = compare(t4, TABLE4, TABLE4_COLS, "TABLE 4: Pi, HB, EF")
    mismatches += compare(t5, TABLE5, TABLE5_COLS, "TABLE 5: SC, FC (%)")

    print(f"\n{'=' * 100}\nEF definition check (paper value vs ratio-of-means vs mean-of-ratios)\n{'=' * 100}")
    for label in TABLE4:
        print(f"{label:22s} TS {TABLE4[label][4]:5.1f} | {t4.loc[label, 'EF_TS']:6.2f} | "
              f"{ef_alt.loc[label, 'EF_TS_perrun']:6.2f}    "
              f"Bayes {TABLE4[label][9]:5.1f} | {t4.loc[label, 'EF_B']:6.2f} | "
              f"{ef_alt.loc[label, 'EF_B_perrun']:6.2f}")

    print(f"\n{'=' * 100}\nMISMATCHES (tolerance +/- {TOL:.2f})\n{'=' * 100}")
    if not mismatches:
        print("None. Every printed value is reproduced.")
    for table, label, col, printed, computed, diff in mismatches:
        print(f"  {table[:7]}  {label:22s} {col:9s} printed {printed:8.2f}  computed {computed:8.2f}  "
              f"(diff {diff:+.2f})")

    REPRO_DIR.mkdir(parents=True, exist_ok=True)
    out = REPRO_DIR / "tables_4_5_reproduced.csv"
    pd.concat([t4, t5, ef_alt], axis=1).round(3).to_csv(out)
    print(f"\nReproduced values saved to: {out}")


if __name__ == "__main__":
    main()
