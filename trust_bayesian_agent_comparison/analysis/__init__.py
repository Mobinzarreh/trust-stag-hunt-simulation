"""Analysis module for trust-bayesian agent comparison."""
from .sensitivity import SensitivityAnalysisManager
from .monte_carlo import MonteCarloManager
from .batch import run_summary, run_cell, run_cells, welch_ci, paired_ci, sig_mark
from .metrics import (
    agent_coop_rate,
    mutual_coop_rate,
    betrayal_rate,
    failed_collaboration_rate,
    final_decision,
    calculate_payoffs,
    total_payoff,
    time_to_threshold,
    compute_strategy_statistics,
    per_run_metrics,
    OUTCOME_COUNT_COLS,
    outcome_counts,
    payoff_from_counts,
)

__all__ = [
    'SensitivityAnalysisManager',
    'MonteCarloManager',
    'agent_coop_rate',
    'mutual_coop_rate',
    'betrayal_rate',
    'failed_collaboration_rate',
    'final_decision',
    'calculate_payoffs',
    'total_payoff',
    'time_to_threshold',
    'compute_strategy_statistics',
    'per_run_metrics',
    'OUTCOME_COUNT_COLS',
    'outcome_counts',
    'payoff_from_counts',
    'run_summary',
    'run_cell',
    'run_cells',
    'welch_ci',
    'paired_ci',
    'sig_mark',
]
