"""Trust-based Bayesian agent comparison framework."""

from .config import (
    PAYOFF_MATRIX,
    stag_indifference_threshold,
    get_payoff,
    DECISION_THRESHOLD,
)
from .agents import FocalAgent, BayesianFocalAgent, DiscountedBetaAgent, SlidingWindowBetaAgent

__version__ = "0.1.0"

__all__ = [
    "FocalAgent",
    "BayesianFocalAgent",
    "DiscountedBetaAgent",
    "SlidingWindowBetaAgent",
    "PAYOFF_MATRIX",
    "stag_indifference_threshold",
    "get_payoff",
    "DECISION_THRESHOLD",
]
