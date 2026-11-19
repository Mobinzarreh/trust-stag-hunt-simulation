#!/usr/bin/env python3
"""
Robustness of the detection and repair delays to the persistence length L (Appendix B).

The detection delay D_det and the repair delay D_rep (Eqs. 25 and 26) count the rounds from a change in the
partner's collaboration probability until the agent's estimate p-hat has stayed on the new side of u for L
consecutive rounds. The paper uses L = 5 (behavioral_metrics.PERSIST_ESTIMATE). This script re-scores the same
runs for L = 1, ..., 10 and reports, for each L, the mean (SE) delay of every agent and the trust-sensitive agent
minus each benchmark.

Runs: the four agents of the main comparison (trust-sensitive default, cumulative Bayes, discounted Bayes with a
fixed prior, rho = 0.9, sliding window, w = 10) against the four scheduled partners whose collaboration probability
crosses u (Single Cycle, Gradual Deterioration, Break and Repair, Reliability Ramp), 300 runs each, with the seeds
and RNG slots of the main study (run i uses seed MC_BASE_SEED + i; the trust agent in slot 1, benchmarks in slot 2).
Runs in which the condition is not met before the partner's next change, or before the end, are censored (left out
of the mean), as in the paper; their count is reported.

Check: at L = 5 the delays equal those in results/variants/runs_long.csv (when that file exists).

Outputs (results/persistence_length/):
    per_run.csv    agent, partner, run_id, L, delay_down_sust, delay_up_sust
    summary.csv    mean, se, n, censored per agent x partner x L x delay
    contrasts.csv  trust minus each benchmark (diff, se) per partner x L x delay
    report.md      the trust-minus-DiscPP table and the ordering check

Usage:
    python scripts/check_persistence_length.py
"""
import math
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trust_bayesian_agent_comparison.agents import (BayesianFocalAgent, FocalAgent,  # noqa: E402
                                                    PriorPreservingDiscountedBetaAgent, SlidingWindowBetaAgent)
from trust_bayesian_agent_comparison.analysis import run_cell  # noqa: E402
from trust_bayesian_agent_comparison.analysis import behavioral_metrics as bm  # noqa: E402
from trust_bayesian_agent_comparison.config import (DISCOUNTED_BETA_RHO, NUM_MONTE_CARLO_RUNS,  # noqa: E402
                                                    NUM_ROUNDS, RESULTS_DIR, SLIDING_WINDOW_W)
from trust_bayesian_agent_comparison.partners import partner_factory  # noqa: E402

PARTNERS = ("SingleCycle", "GradualDeterioration", "BreakRepair", "ReliabilityRamp")
# key -> (label, factory, RNG slot); keys follow results/variants (B = the trust-sensitive default)
AGENTS = {
    "B": ("TS", FocalAgent, 1),
    "discpp0.9": ("Disc", partial(PriorPreservingDiscountedBetaAgent, rho=DISCOUNTED_BETA_RHO), 2),
    "win10": ("SW", partial(SlidingWindowBetaAgent, window=SLIDING_WINDOW_W), 2),
    "cumulative": ("Cum", BayesianFocalAgent, 2),
}
LENGTHS = range(1, 11)
DELAYS = ("delay_down_sust", "delay_up_sust")
OUT = RESULTS_DIR / "persistence_length"


def score(key: str, partner: str) -> pd.DataFrame:
    """Simulate one cell and score both delays for every L."""
    _, factory, slot = AGENTS[key]
    _, traj = run_cell(factory, partner_factory(partner, NUM_ROUNDS), slot,
                       num_runs=NUM_MONTE_CARLO_RUNS, return_trajectories=True)
    tr = bm.make_trace(traj, partner_factory(partner, NUM_ROUNDS)())
    default = bm.PERSIST_ESTIMATE
    frames = []
    try:
        for L in LENGTHS:
            bm.PERSIST_ESTIMATE = L          # per_run reads the module constant at call time
            pr = bm.per_run(tr)
            frames.append(pd.DataFrame({"agent": key, "partner": partner, "run_id": np.arange(tr.n), "L": L,
                                        **{d: pr[d].to_numpy() for d in DELAYS}}))
    finally:
        bm.PERSIST_ESTIMATE = default
    return pd.concat(frames, ignore_index=True)


def check_against_variants(per_run: pd.DataFrame) -> None:
    src = RESULTS_DIR / "variants" / "runs_long.csv"
    if not src.exists():
        print("results/variants/runs_long.csv not found; L = 5 check skipped.")
        return
    ref = pd.read_csv(src, low_memory=False, usecols=["partner", "agent", "run_id", *DELAYS])
    ours = per_run[per_run.L == 5].drop(columns="L")
    m = ours.merge(ref, on=["agent", "partner", "run_id"], suffixes=("", "_ref"), validate="one_to_one")
    assert len(m) == len(ours), "some runs have no counterpart in runs_long.csv"
    for d in DELAYS:
        a, b = m[d].to_numpy(float), m[f"{d}_ref"].to_numpy(float)
        assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)]), d
    print(f"L = 5 equals results/variants/runs_long.csv on {len(m)} runs (both delays, censoring included).")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = [(k, p) for k in AGENTS for p in PARTNERS]
    per_run = pd.concat(Parallel(n_jobs=-1)(delayed(score)(k, p) for k, p in cells), ignore_index=True)
    check_against_variants(per_run)
    per_run.to_csv(OUT / "per_run.csv", index=False)

    long = per_run.melt(id_vars=["agent", "partner", "run_id", "L"], value_vars=list(DELAYS),
                        var_name="delay", value_name="rounds")
    g = long.groupby(["agent", "partner", "L", "delay"])["rounds"]
    summary = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n": g.count(),
                            "censored": g.apply(lambda s: int(s.isna().sum()))}).reset_index()
    summary["se"] = summary["sd"] / np.sqrt(summary["n"])
    # a delay exists only where the schedule crosses u in that direction
    summary = summary[summary["n"] > 0]
    summary.to_csv(OUT / "summary.csv", index=False)

    s = summary.set_index(["agent", "partner", "L", "delay"])
    rows = []
    ts = summary[summary.agent == "B"]
    for _, r in ts.iterrows():
        for bench in ("discpp0.9", "win10", "cumulative"):
            idx = (bench, r.partner, r.L, r.delay)
            if idx not in s.index:
                continue
            b = s.loc[idx]
            rows.append({"partner": r.partner, "L": r.L, "delay": r.delay, "benchmark": bench,
                         "diff": r["mean"] - b["mean"], "se": math.sqrt(r.se ** 2 + b.se ** 2),
                         "censored_ts": r.censored, "censored_bench": int(b.censored)})
    con = pd.DataFrame(rows)
    con.to_csv(OUT / "contrasts.csv", index=False)

    # Ordering check: the sign of trust minus each benchmark is the one it has at the paper's L = 5
    ref = con[con.L == 5].set_index(["partner", "delay", "benchmark"])["diff"]
    con["as_at_L5"] = [np.sign(r["diff"]) == np.sign(ref[(r.partner, r.delay, r.benchmark)]) for _, r in con.iterrows()]
    lines = ["# Persistence length L of the detection and repair delays", "",
             "Trust-sensitive agent minus discounted Bayes (rho = 0.9), mean (SE) in rounds, 300 runs per agent.", ""]
    d = con[con.benchmark == "discpp0.9"]
    for delay, name in (("delay_down_sust", "Detection delay"), ("delay_up_sust", "Repair delay")):
        t = d[d.delay == delay].pivot(index="L", columns="partner", values=["diff", "se"])
        lines += [f"## {name}", "", "| L | " + " | ".join(t["diff"].columns) + " |",
                  "|---|" + "---|" * len(t["diff"].columns)]
        for L in t.index:
            cells_ = [f"{t['diff'].loc[L, p]:+.2f} ({t['se'].loc[L, p]:.2f})" for p in t["diff"].columns]
            lines.append(f"| {L} | " + " | ".join(cells_) + " |")
        lines.append("")
    for bench in ("discpp0.9", "win10", "cumulative"):
        sub = con[con.benchmark == bench]
        lines.append(f"- vs {AGENTS[bench][0]}: same direction as at L = 5 in {int(sub.as_at_L5.sum())} of "
                     f"{len(sub)} partner x L x delay cells.")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
