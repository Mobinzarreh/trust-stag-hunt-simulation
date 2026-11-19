"""Agent implementations for trust-based and Bayesian learning."""

from .base import BaseAgent
from .focal_agent import FocalAgent
from .bayesian_agent import BayesianFocalAgent
from .discounted_beta_agent import DiscountedBetaAgent
from .sliding_window_agent import SlidingWindowBetaAgent
from .discounted_beta_agent import PriorPreservingDiscountedBetaAgent
from .changepoint_agent import ChangePointBetaAgent
from .thompson_agents import (
    ThompsonChoiceMixin,
    ThompsonCumulativeBetaAgent,
    ThompsonDiscountedBetaAgent,
    ThompsonWindowBetaAgent,
)

__all__ = ['BaseAgent', 'FocalAgent', 'BayesianFocalAgent', 'DiscountedBetaAgent', 'SlidingWindowBetaAgent',
           'PriorPreservingDiscountedBetaAgent', 'ChangePointBetaAgent', 'ThompsonChoiceMixin',
           'ThompsonCumulativeBetaAgent', 'ThompsonDiscountedBetaAgent', 'ThompsonWindowBetaAgent']
