"""Beta benchmarks whose posterior uncertainty reaches the action rule (Thompson sampling)."""

import numpy as np

from .bayesian_agent import BayesianFocalAgent
from .discounted_beta_agent import DiscountedBetaAgent
from .sliding_window_agent import SlidingWindowBetaAgent


class ThompsonChoiceMixin:
    """
    Replaces the shared logit rule with Thompson sampling over the agent's own Beta posterior.

    Every other agent in this study, including the trust-sensitive one, chooses with
        P(C) = sigma(theta * (EV_C(p_hat) - EV_D(p_hat)))
    evaluated at a point estimate p_hat, so a Beta benchmark's concentration alpha + beta never
    reaches its decisions: two agents with the same mean act identically however much evidence
    each has. The trust agent is the only one whose confidence is behaviorally active, because
    t_k shrinks p_hat toward 0.5. That is an asymmetry in the comparison, not only in the model.

    Thompson sampling removes it. The agent draws
        p_tilde ~ Beta(alpha, beta)
    and best-responds to the draw. A diffuse posterior then produces variable, exploratory
    behavior and a concentrated one produces near-deterministic behavior, with no logit noise
    added on top. This is also the published form of the sliding-window benchmark
    (Trovo et al. 2020 specify sliding-window Thompson sampling, not a logit rule).

    Trade-off to state if this is reported: the action rule is no longer shared across agents,
    so these variants are a robustness check on the role of confidence rather than part of the
    memory-matched family, and theta does not apply to them.
    """

    def decide(self) -> int:
        """
        Sample a collaboration probability from the posterior, then best-respond to it.

        Returns:
            int: 1 for collaborate, 0 for independence
        """
        M = self.payoff_matrix
        p = float(np.random.beta(self.alpha, self.beta))

        EV_defect = (1 - p) * M[0, 0, 0] + p * M[0, 1, 0]
        EV_cooperate = (1 - p) * M[1, 0, 0] + p * M[1, 1, 0]

        return 1 if EV_cooperate > EV_defect else 0


class ThompsonCumulativeBetaAgent(ThompsonChoiceMixin, BayesianFocalAgent):
    """Cumulative Beta-Bernoulli benchmark, Thompson sampling instead of the logit rule."""


class ThompsonDiscountedBetaAgent(ThompsonChoiceMixin, DiscountedBetaAgent):
    """Discounted Beta benchmark (rho), Thompson sampling instead of the logit rule."""


class ThompsonWindowBetaAgent(ThompsonChoiceMixin, SlidingWindowBetaAgent):
    """Sliding-window Beta benchmark (w), Thompson sampling: the SW-TS of Trovo et al. (2020)."""
