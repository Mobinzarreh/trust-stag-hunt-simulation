"""Dynamic Bayesian benchmark: Beta-Bernoulli with online change-point detection."""

import numpy as np

from .bayesian_agent import BayesianFocalAgent
from ..config import EPS, STOCHASTIC, INVERSE_TEMPERATURE, CHANGEPOINT_HAZARD


class ChangePointBetaAgent(BayesianFocalAgent):
    """
    Bayesian online change-point detection (Adams & MacKay 2007) over a Beta-Bernoulli
    likelihood. This is the "dynamic Bayesian" benchmark: it drops the stationarity
    assumption without giving up exact Bayesian updating.

    Instead of one Beta posterior, the agent carries a posterior over the run length r
    (rounds since the partner last changed regime) together with a Beta(eps, eps)
    posterior conditional on each run length. With hazard h (the prior probability that
    any round starts a new regime) and pi_r = a_r / (a_r + b_r):

        likelihood   L_r = pi_r if o_k = 1 else 1 - pi_r
        growth       w_k(r + 1) ∝ (1 - h) * w_{k-1}(r) * L_r
        change       w_k(0)     ∝ h * sum_r w_{k-1}(r) * L_r
        counts       a_{r+1} = a_r + o_k,  b_{r+1} = b_r + (1 - o_k),  a_0, b_0 = eps, eps

    Estimate: the posterior predictive mean, mixed over run lengths,

        p_hat = sum_r w(r) * pi_r

    h = 0 reproduces BayesianFocalAgent bit for bit (no regime change is ever possible).
    h = 1 - rho matches the discounted model's expected memory of 1 / h rounds, so the
    primary setting h = 0.1 is memory-matched to rho = 0.9 and w = 10.

    Why this benchmark is not just another forgetting rule: discounting and a window shed
    the past at a fixed rate whatever the partner does, so their concentration is a
    deterministic function of the round. Here the effective sample size
    sum_r w(r) (a_r + b_r) is data-dependent. It collapses when an observation is
    surprising under the current regime and rebuilds while the partner is predictable,
    which is the Bayesian counterpart of the trust agent's confidence state.
    """

    def __init__(
        self,
        hazard: float = CHANGEPOINT_HAZARD,
        alpha0: float = EPS,
        beta0: float = EPS,
        stochastic: bool = STOCHASTIC,
        inv_temp: float = INVERSE_TEMPERATURE,
        payoff_matrix=None,
    ):
        """
        Initialize change-point Beta agent.

        Args:
            hazard: Per-round prior probability of a regime change, h in [0, 1)
            alpha0: Prior pseudo-count for cooperation, used for every fresh regime
            beta0: Prior pseudo-count for defection, used for every fresh regime
            stochastic: If True, use logit choice; if False, deterministic threshold
            inv_temp: Inverse temperature for logit function
            payoff_matrix: Payoff matrix for decisions (default: config.PAYOFF_MATRIX)
        """
        if not 0 <= hazard < 1:
            raise ValueError("hazard must be in [0, 1)")
        super().__init__(alpha0=alpha0, beta0=beta0, stochastic=stochastic, inv_temp=inv_temp,
                         payoff_matrix=payoff_matrix)
        self.hazard = float(hazard)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)

        # Run-length posterior w(r) and the Beta counts conditional on each run length
        self.w = np.array([1.0])
        self.a = np.array([self.alpha0])
        self.b = np.array([self.beta0])
        self.p_hat = self.alpha0 / (self.alpha0 + self.beta0)

        # Expected run length, the model's own "how much history still counts" measure
        self.history_runlength = [0.0]

    def expected_p(self) -> float:
        """
        Posterior predictive mean, mixed over the run-length posterior.

        Returns:
            float: sum_r w(r) * a_r / (a_r + b_r)
        """
        return self.p_hat

    def update(self, partner_choice: int):
        """
        One step of the forward recursion: reweight run lengths, then extend the counts.

        Args:
            partner_choice: Partner's action (1=cooperate, 0=defect)
        """
        if partner_choice not in (0, 1):
            raise ValueError("partner_choice must be 0 or 1")

        o = int(partner_choice)
        h = self.hazard

        # Predictive likelihood of this observation under each run length
        pi = self.a / (self.a + self.b)
        evidence = self.w * (pi if o == 1 else 1.0 - pi)

        # Change-point mass collapses to r = 0; the rest grows by one round
        w_new = np.empty(self.w.size + 1)
        w_new[0] = h * evidence.sum()
        w_new[1:] = (1.0 - h) * evidence
        self.w = w_new / w_new.sum()

        # Each surviving run length absorbs the observation; r = 0 restarts from the prior
        self.a = np.concatenate(([self.alpha0], self.a + o))
        self.b = np.concatenate(([self.beta0], self.b + (1 - o)))

        self.p_hat = float((self.w * (self.a / (self.a + self.b))).sum())

        # Report the mixture as an equivalent Beta so alpha, beta and their sum stay
        # comparable with the other benchmarks: the mean is p_hat and the concentration is
        # the expected effective sample size.
        ess = float((self.w * (self.a + self.b)).sum())
        self.alpha = self.p_hat * ess
        self.beta = (1.0 - self.p_hat) * ess

        # Record history
        self.history_alpha.append(self.alpha)
        self.history_beta.append(self.beta)
        self.history_runlength.append(float((self.w * np.arange(self.w.size)).sum()))
