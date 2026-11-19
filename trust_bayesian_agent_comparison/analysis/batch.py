"""
Batch Monte Carlo runs that keep one summary row per run instead of the full trajectory.

The B1-B4 scripts store every round of every run through MonteCarloManager. Sweeps with
hundreds of cells (game structure, one-factor-at-a-time sensitivity) only need per-run
metrics, so this module runs one agent in one RNG slot and summarizes each run.

A cell here reproduces MonteCarloManager exactly: run i uses seed base_seed + i and the slot's
RNG stream (slot 1 = agent1, slot 2 = agent2 of run_paired_simulation), so its metrics equal
per_run_metrics() of the corresponding stored CSV.

Also holds the two confidence-interval helpers the revision scripts share.
"""

from typing import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..config import MC_BASE_SEED, NUM_MONTE_CARLO_RUNS, NUM_ROUNDS, N_JOBS, best_response_payoff
from ..simulation import run_agent_simulation, slot_rng
from .metrics import outcome_counts

# Same normal quantile as the B1-B4 scripts, so significance marks agree with their reports
Z95 = 1.96


def run_summary(df: pd.DataFrame, payoff_matrix) -> dict:
    """
    One run's summary: Pi, HB, SC, FC (manuscript Eqs. 17-20), the joint-outcome counts, and the
    mean confidence t over rounds for agents that have one.
    """
    a = df["agent_action"].to_numpy()
    o = df["partner_action"].to_numpy()
    row = {
        "Pi": float(df["agent_payoff"].sum()),
        "HB": float(best_response_payoff(payoff_matrix)[o].sum()),
        "SC": float(((a == 1) & (o == 1)).mean()),
        "FC": float(((a == 1) & (o == 0)).mean()),
        **outcome_counts(a, o),
    }
    row["t_mean"] = float(df["trust_level"].mean()) if "trust_level" in df else np.nan
    return row


def run_cell(
    agent_factory: Callable,
    partner_factory: Callable,
    slot: int,
    num_runs: int = NUM_MONTE_CARLO_RUNS,
    base_seed: int = MC_BASE_SEED,
    num_rounds: int = NUM_ROUNDS,
    return_trajectories: bool = False,
):
    """
    Run one agent against fresh partners for num_runs runs in RNG slot 1 or 2.

    The game is the agent's own payoff_matrix. Returns a DataFrame with one row per run_id, and,
    if return_trajectories, also the concatenated round-level frames (with run_id).
    """
    rows, trajectories = [], []
    for i in range(num_runs):
        seed = base_seed + i
        agent = agent_factory()
        df = run_agent_simulation(agent, partner_factory(), num_rounds, seed, rng=slot_rng(seed, slot))
        rows.append({"run_id": i, **run_summary(df, agent.payoff_matrix)})
        if return_trajectories:
            df["run_id"] = i
            trajectories.append(df)
    summary = pd.DataFrame(rows)
    if return_trajectories:
        return summary, pd.concat(trajectories, ignore_index=True)
    return summary


def run_cells(cells: dict, n_jobs: int = N_JOBS, verbose: int = 0) -> dict:
    """
    Run many cells in parallel.

    Args:
        cells: {key: dict of run_cell keyword arguments}; factories must be picklable
        n_jobs: joblib workers (-1 = all cores)

    Returns:
        {key: per-run DataFrame}
    """
    keys = list(cells)
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(run_cell)(**cells[k]) for k in keys)
    return dict(zip(keys, results))


def welch_ci(a, b, scale: float = 1.0) -> tuple[float, float, float]:
    """Difference of means scale * (a - b) with an unpaired (Welch) 95% CI, for independent streams."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = scale * (a.mean() - b.mean())
    se = abs(scale) * np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, d - Z95 * se, d + Z95 * se


def paired_ci(a, b, scale: float = 1.0) -> tuple[float, float, float]:
    """Mean of scale * (a - b) over matched runs with a paired 95% CI (same RNG stream, same run_id)."""
    d = scale * (np.asarray(a, float) - np.asarray(b, float))
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), d.mean() - Z95 * se, d.mean() + Z95 * se


def sig_mark(lo: float, hi: float) -> str:
    """'+' if the CI is above 0, '-' if below, '' if it contains 0."""
    return "+" if lo > 0 else ("-" if hi < 0 else "")
