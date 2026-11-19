"""Beta-Bernoulli benchmark with exponential forgetting (Beta reputation with decay)."""

from .bayesian_agent import BayesianFocalAgent
from ..config import EPS, STOCHASTIC, INVERSE_TEMPERATURE, DISCOUNTED_BETA_RHO


class DiscountedBetaAgent(BayesianFocalAgent):
    """
    Beta reputation model with a forgetting factor rho.

    Same Beta(eps, eps) start, posterior-mean estimate and decision rule as
    BayesianFocalAgent. Only the memory changes:
    - alpha <- rho * alpha + o
    - beta  <- rho * beta  + (1 - o)
    - Belief: E[θ] = α / (α + β)

    rho = 1 reproduces BayesianFocalAgent exactly. Smaller rho forgets faster.
    Following Parhizkar et al. (2022, Eq. 2.1), the prior pseudo-counts are discounted
    together with the evidence, so the prior's weight decays as rho**k and the
    concentration alpha + beta tends to 1 / (1 - rho).
    """

    def __init__(
        self,
        rho: float = DISCOUNTED_BETA_RHO,
        alpha0: float = EPS,
        beta0: float = EPS,
        stochastic: bool = STOCHASTIC,
        inv_temp: float = INVERSE_TEMPERATURE,
        payoff_matrix=None,
    ):
        """
        Initialize discounted Beta agent.

        Args:
            rho: Forgetting factor in (0, 1]
            alpha0: Prior pseudo-count for cooperation (α₀)
            beta0: Prior pseudo-count for defection (β₀)
            stochastic: If True, use logit choice; if False, deterministic threshold
            inv_temp: Inverse temperature for logit function
            payoff_matrix: Payoff matrix for decisions (default: config.PAYOFF_MATRIX)
        """
        if not 0 < rho <= 1:
            raise ValueError("rho must be in (0, 1]")
        super().__init__(alpha0=alpha0, beta0=beta0, stochastic=stochastic, inv_temp=inv_temp,
                         payoff_matrix=payoff_matrix)
        self.rho = float(rho)

    def update(self, partner_choice: int):
        """
        Discount the current counts, then add the new observation.

        Args:
            partner_choice: Partner's action (1=cooperate, 0=defect)
        """
        if partner_choice not in (0, 1):
            raise ValueError("partner_choice must be 0 or 1")

        self.alpha = self.rho * self.alpha + partner_choice
        self.beta = self.rho * self.beta + (1 - partner_choice)

        # Record history
        self.history_alpha.append(self.alpha)
        self.history_beta.append(self.beta)


class PriorPreservingDiscountedBetaAgent(DiscountedBetaAgent):
    """
    Discounted Beta model that forgets the evidence but keeps the Beta(eps, eps) prior fixed.

    DiscountedBetaAgent follows Parhizkar et al. (2022, Eq. 2.1) and decays the prior along
    with the evidence, so its prior weight vanishes as rho**k and its estimate can reach 0 or 1.
    The cumulative model, the window and the trust agent all keep that regularization and are
    bounded. This variant discounts only the evidence counts:

        s <- rho * s + o,        f <- rho * f + (1 - o)
        alpha = alpha0 + s,      beta = beta0 + f
        Belief: E[θ] = (alpha0 + s) / (alpha0 + beta0 + s + f)

    Same memory as DiscountedBetaAgent (geometric, effective length 1 / (1 - rho)), but the
    concentration tends to alpha0 + beta0 + 1 / (1 - rho) = 12 at rho = 0.9, exactly the window's
    and the trust agent's ceiling. Differences from the window are then window shape alone, and
    differences from the trust agent are the trust mechanism alone. rho = 1 still reproduces
    BayesianFocalAgent bit for bit.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Evidence counts, discounted; the prior is held out of the recursion
        self.s = 0.0
        self.f = 0.0
        self.alpha0 = float(self.alpha)
        self.beta0 = float(self.beta)

    def update(self, partner_choice: int):
        """
        Discount the evidence counts only, then rebuild alpha and beta on the fixed prior.

        Args:
            partner_choice: Partner's action (1=cooperate, 0=defect)
        """
        if partner_choice not in (0, 1):
            raise ValueError("partner_choice must be 0 or 1")

        self.s = self.rho * self.s + partner_choice
        self.f = self.rho * self.f + (1 - partner_choice)
        self.alpha = self.alpha0 + self.s
        self.beta = self.beta0 + self.f

        # Record history
        self.history_alpha.append(self.alpha)
        self.history_beta.append(self.beta)
