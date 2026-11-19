"""
Metrics and analysis utilities for simulation results.
"""
import pandas as pd
import numpy as np
from typing import Optional

from ..config import DECISION_THRESHOLD, PAYOFF_MATRIX, as_payoff_matrix, best_response_payoff


def get_payoff(player1_strategy: int, player2_strategy: int, player_id: int, payoff_matrix=None) -> float:
    """Extract payoff from matrix."""
    m = PAYOFF_MATRIX if payoff_matrix is None else payoff_matrix
    return float(m[int(player1_strategy), int(player2_strategy), int(player_id)])


def agent_coop_rate(df: pd.DataFrame) -> float:
    """Calculate fraction of rounds the agent chose to cooperate."""
    return float(df["agent_action"].mean())


def mutual_coop_rate(df: pd.DataFrame) -> float:
    """Calculate fraction of rounds with mutual cooperation."""
    return float(((df["agent_action"] == 1) & (df["partner_action"] == 1)).mean())


def failed_collaboration_rate(df: pd.DataFrame) -> float:
    """
    Fraction of rounds where the agent collaborated but the partner chose independence.

    This is a realized joint outcome. It is distinct from the expectation violations
    used inside FocalAgent's trust update, which are prediction errors that do not
    depend on the agent's own action.
    """
    return float(((df["agent_action"] == 1) & (df["partner_action"] == 0)).mean())


def betrayal_rate(df: pd.DataFrame) -> float:
    """Alias of failed_collaboration_rate, kept for backward compatibility."""
    return failed_collaboration_rate(df)


def final_decision(df: pd.DataFrame) -> int:
    """Return final decision (0 or 1)."""
    return int(df["agent_action"].iloc[-1])


def calculate_payoffs(df: pd.DataFrame, payoff_matrix=None) -> pd.DataFrame:
    """Add payoff columns to simulation results."""
    df = df.copy()
    
    agent_payoffs = []
    partner_payoffs = []
    
    for _, row in df.iterrows():
        agent_choice = int(row["agent_action"])
        partner_choice = int(row["partner_action"])
        
        agent_payoffs.append(get_payoff(agent_choice, partner_choice, 0, payoff_matrix))
        partner_payoffs.append(get_payoff(agent_choice, partner_choice, 1, payoff_matrix))
    
    df["agent_payoff"] = agent_payoffs
    df["partner_payoff"] = partner_payoffs
    
    return df


def total_payoff(df: pd.DataFrame) -> float:
    """Calculate total agent payoff across all rounds."""
    if "agent_payoff" not in df.columns:
        df = calculate_payoffs(df)
    return float(df["agent_payoff"].sum())


def average_payoff(df: pd.DataFrame) -> float:
    """Calculate average agent payoff per round (scale-independent)."""
    if "agent_payoff" not in df.columns:
        df = calculate_payoffs(df)
    return float(df["agent_payoff"].mean())


BEST_RESPONSE_PAYOFF = best_response_payoff()


def per_run_metrics(df: pd.DataFrame, payoff_matrix=None) -> pd.DataFrame:
    """
    Manuscript metrics for each Monte Carlo run in a results DataFrame.

    Returns one row per run_id with:
        Pi  total payoff                       (Eq. 17)
        HB  hindsight benchmark                (Eq. 20)
        SC  successful collaboration rate      (Eq. 18)
        FC  failed collaboration rate          (Eq. 19)

    Pi is read from the recorded payoffs; payoff_matrix must be the game those runs were played in.
    """
    a = df["agent_action"].to_numpy()
    o = df["partner_action"].to_numpy()
    best = BEST_RESPONSE_PAYOFF if payoff_matrix is None else best_response_payoff(payoff_matrix)
    tmp = pd.DataFrame({
        "run_id": df["run_id"].to_numpy(),
        "Pi": df["agent_payoff"].to_numpy(),
        "HB": best[o],
        "SC": ((a == 1) & (o == 1)).astype(float),
        "FC": ((a == 1) & (o == 0)).astype(float),
    })
    return tmp.groupby("run_id").agg(Pi=("Pi", "sum"), HB=("HB", "sum"),
                                     SC=("SC", "mean"), FC=("FC", "mean"))


OUTCOME_COUNT_COLS = ("n_CC", "n_CI", "n_IC", "n_II")


def outcome_counts(agent_action, partner_action) -> dict:
    """Counts of (agent, partner) = (C, C), (C, I), (I, C), (I, I) over one run; C = 1, I = 0."""
    a = np.asarray(agent_action)
    o = np.asarray(partner_action)
    return {
        "n_CC": int(((a == 1) & (o == 1)).sum()),
        "n_CI": int(((a == 1) & (o == 0)).sum()),
        "n_IC": int(((a == 0) & (o == 1)).sum()),
        "n_II": int(((a == 0) & (o == 0)).sum()),
    }


def payoff_from_counts(counts, payoff_matrix=None):
    """Focal-agent total payoff from outcome counts (dict, Series or DataFrame columns n_CC ... n_II)."""
    m = as_payoff_matrix(payoff_matrix)
    return (counts["n_CC"] * m[1, 1, 0] + counts["n_CI"] * m[1, 0, 0]
            + counts["n_IC"] * m[0, 1, 0] + counts["n_II"] * m[0, 0, 0])


def time_to_threshold(
    df: pd.DataFrame,
    p_star: Optional[float] = None,
    direction: Optional[str] = None,
    start_round: int = 0,
) -> Optional[int]:
    """
    First round, at or after start_round, in which E[p] crosses the threshold.

    "Above" means E[p] >= p_star and "below" means E[p] < p_star, matching the decision rule's
    indifference point. A crossing in round i means round i is on the other side from round i - 1,
    so an estimate that starts below the threshold has not crossed down in round 0.

    Args:
        df: Simulation results for one run, ordered by round
        p_star: Threshold value (default: DECISION_THRESHOLD)
        direction: 'up' (below to above), 'down' (above to below), or None (either way)
        start_round: Only crossings in rounds >= start_round count (e.g. a known change point)

    Returns:
        Round number as stored in df['round'] (0-indexed), or None if there is no such crossing
    """
    if p_star is None:
        p_star = DECISION_THRESHOLD
    if direction not in ("up", "down", None):
        raise ValueError("direction must be 'up', 'down' or None")

    E = df["agent_belief"].to_numpy()
    rounds = df["round"].to_numpy()
    if len(E) < 2:
        return None

    above = E >= p_star
    changed = above[1:] != above[:-1]
    if direction == "up":
        changed &= above[1:]
    elif direction == "down":
        changed &= ~above[1:]
    idx = np.flatnonzero(changed & (rounds[1:] >= start_round)) + 1
    return int(rounds[idx[0]]) if idx.size else None


def compute_strategy_statistics(df: pd.DataFrame) -> dict:
    """
    Compute comprehensive statistics for a single simulation.
    
    Returns dictionary with:
        - Cooperation rates
        - Payoff statistics
        - Final beliefs
        - Threshold metrics
    """
    df_with_payoffs = calculate_payoffs(df)
    
    stats = {
        "agent_coop_rate": agent_coop_rate(df),
        "mutual_coop_rate": mutual_coop_rate(df),
        "betrayal_rate": betrayal_rate(df),
        "total_agent_payoff": total_payoff(df_with_payoffs),
        "avg_agent_payoff": df_with_payoffs["agent_payoff"].mean(),
        "total_partner_payoff": df_with_payoffs["partner_payoff"].sum(),
        "avg_partner_payoff": df_with_payoffs["partner_payoff"].mean(),
        "final_decision": final_decision(df),
        "final_belief": float(df["agent_belief"].iloc[-1]),
    }
    
    return stats
