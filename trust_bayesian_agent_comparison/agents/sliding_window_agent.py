"""Beta-Bernoulli benchmark over a sliding window of recent observations."""

from collections import deque

from .bayesian_agent import BayesianFocalAgent
from ..config import EPS, STOCHASTIC, INVERSE_TEMPERATURE, SLIDING_WINDOW_W


class SlidingWindowBetaAgent(BayesianFocalAgent):
    """
    Beta-smoothed moving average of the partner's last w actions.

    Same Beta(eps, eps) start, posterior-mean estimate and decision rule as
    BayesianFocalAgent. Only the memory changes. After k observations, with
    n = min(k, w) of them in the window and s of those collaborative:
    - alpha = alpha0 + s
    - beta  = beta0 + n - s
    - Belief: E[θ] = (alpha0 + s) / (alpha0 + beta0 + n)

    An observation that leaves the window has no further influence. w >= the number
    of rounds reproduces BayesianFocalAgent exactly. This is the windowed Beta of
    Trovò et al. (2020); the action rule stays the shared logit rule, not Thompson sampling.
    """

    def __init__(
        self,
        window: int = SLIDING_WINDOW_W,
        alpha0: float = EPS,
        beta0: float = EPS,
        stochastic: bool = STOCHASTIC,
        inv_temp: float = INVERSE_TEMPERATURE,
        payoff_matrix=None,
    ):
        """
        Initialize sliding-window Beta agent.

        Args:
            window: Number of most recent observations retained (w >= 1)
            alpha0: Prior pseudo-count for cooperation (α₀)
            beta0: Prior pseudo-count for defection (β₀)
            stochastic: If True, use logit choice; if False, deterministic threshold
            inv_temp: Inverse temperature for logit function
            payoff_matrix: Payoff matrix for decisions (default: config.PAYOFF_MATRIX)
        """
        if int(window) != window or window < 1:
            raise ValueError("window must be a positive integer")
        super().__init__(alpha0=alpha0, beta0=beta0, stochastic=stochastic, inv_temp=inv_temp,
                         payoff_matrix=payoff_matrix)
        self.window = int(window)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)
        self.recent = deque(maxlen=self.window)

    def update(self, partner_choice: int):
        """
        Push the new observation into the window and recount.

        Args:
            partner_choice: Partner's action (1=cooperate, 0=defect)
        """
        if partner_choice not in (0, 1):
            raise ValueError("partner_choice must be 0 or 1")

        self.recent.append(int(partner_choice))
        s = sum(self.recent)
        n = len(self.recent)
        self.alpha = self.alpha0 + s
        self.beta = self.beta0 + (n - s)

        # Record history
        self.history_alpha.append(self.alpha)
        self.history_beta.append(self.beta)
