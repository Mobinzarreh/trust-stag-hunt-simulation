"""Trust-based focal agent with dual-state representation."""

import numpy as np
from .base import BaseAgent
from ..config import (
    TRUST_MIN,
    TRUST_MAX,
    LOSS_AVERSION,
    MEMORY_DISCOUNT,
    TRUST_DISCOUNT,
    TRUST_SMOOTHING,
    ETA,
    NOISE_SIGMA,
    STOCHASTIC,
    INVERSE_TEMPERATURE,
    EPS,
    LAMBDA_SURPRISE,
    stag_indifference_threshold,
    AGENT_LOSS_AVERSION,
    EXPOSED_BETRAYAL_WEIGHT,
    ATTRIBUTION,
    CONFIDENCE_SCORING,
    AGENT_LAMBDA_SURPRISE,
    AGENT_ETA,
    AGENT_T_MAX,
    SURPRISE_SCALE,
)


class FocalAgent(BaseAgent):
    """
    Trust-sensitive agent of the paper: a belief (signal) x about the partner's collaboration and a confidence
    (trust) t in that belief. The decision stage reads p-hat = (eps + x t) / (2 eps + t).

    Default specification (the revised model reported in the paper):
      * x is a recency-weighted average of the partner's actions (discount memory_discount, step eta). The three
        outcome types are weights on the evidence in x: a match weighs 1; a failed collaboration (the agent
        collaborated, the partner defected) weighs exposed_betrayal_weight nu = 3; a surprise (the agent held back,
        the partner collaborated) weighs psi(u) = surprise_scale (1 - u) / u, with u the stag indifference
        threshold of the agent's own payoff matrix (psi = 0.75 at the study game u = 2/3).
      * attribution: nu applies only to an unprovoked failed collaboration. A defection in the round after the
        agent's own defection (a reciprocator's reply) weighs 1.
      * t is a smoothed, discounted mean of the probability the belief gave to what happened
        (confidence_scoring='graded'), bounded by t_min and t_max = 10, so it rises when predictions succeed and
        falls when they fail.
    Every argument left at None resolves to this default; an explicit value always wins.

    submitted_v1=True restores the originally submitted specification exactly (betrayal penalty lambda = 3 and
    surprise weight mu = 0.5 in the confidence update, nu = psi = 1, hard 0/1 consistency, eta = 0.7, t_max = 10).
    """
    
    def __init__(
        self,
        u_i: float = None,
        t_init: float = 0.0,
        eta: float = None,
        noise_sigma: float = NOISE_SIGMA,
        memory_discount: float = MEMORY_DISCOUNT,
        trust_discount: float = TRUST_DISCOUNT,
        trust_smoothing: float = TRUST_SMOOTHING,
        stochastic: bool = STOCHASTIC,
        inv_temp: float = INVERSE_TEMPERATURE,
        loss_aversion: float = None,
        lambda_surprise: float = None,
        t_min: float = TRUST_MIN,
        t_max: float = None,
        eps: float = EPS,
        payoff_matrix=None,
        exposed_betrayal_weight: float = None,
        attribution: bool = None,
        confidence_scoring: str = None,
        surprise_weight: float = None,
        surprise_scale: float = None,
        submitted_v1: bool = False,
    ):
        """
        Initialize focal agent.
        
        Args:
            u_i: Initial signal value (if None, uses 1 - u of payoff_matrix)
            t_init: Initial trust level
            eta: Learning rate for signal update (default config.AGENT_ETA = 0.85; 0.7 under submitted_v1)
            noise_sigma: Gaussian noise std for signal exploration
            memory_discount: Discount factor for signal history
            trust_discount: Discount factor for trust history
            trust_smoothing: Smoothing factor for trust convergence
            stochastic: If True, use logit choice; if False, deterministic threshold
            inv_temp: Inverse temperature for logit function
            loss_aversion: λ ≥ 1: betrayal penalty multiplier in the consistency measure
                (default config.AGENT_LOSS_AVERSION = 1; 3 under submitted_v1)
            lambda_surprise: μ ∈ [0,λ]: surprise penalty weight in the consistency measure
                (default config.AGENT_LAMBDA_SURPRISE = 1; 0.5 under submitted_v1)
            t_min: Lower bound on trust (confidence)
            t_max: Upper bound on trust (confidence); default config.AGENT_T_MAX = 10
                (config.TRUST_MAX = 10 under submitted_v1)
            eps: Beta baseline pseudo-count (eps=1 gives a Beta(1,1) baseline)
            payoff_matrix: Payoff matrix for decisions (default: config.PAYOFF_MATRIX)
            exposed_betrayal_weight: nu >= 1, weight of an exposed betrayal in the belief memory
                (default config.EXPOSED_BETRAYAL_WEIGHT = 3; 1 under submitted_v1)
            attribution: weight only unprovoked exposed betrayals (default True; irrelevant when nu = 1)
            confidence_scoring: 'graded' (default) or 'hit' (the submitted hard 0/1 expectation)
            surprise_weight: psi > 0, weight of a surprise in the belief memory. None resolves to
                surprise_scale * (1 - u) / u with u the stag indifference threshold of the agent's payoff matrix
                (0.75 at the study game); 1 under submitted_v1
            surprise_scale: c in the rule above (default config.SURPRISE_SCALE = 1.5); ignored under submitted_v1 and
                whenever surprise_weight is explicit
            submitted_v1: reproduce the submitted specification (lambda = 3, mu = 0.5, nu = 1, psi = 1, 'hit',
                t_max = 10) for every argument left at None
        """
        spec = (dict(loss_aversion=LOSS_AVERSION, exposed_betrayal_weight=1.0, attribution=False,
                     confidence_scoring="hit", lambda_surprise=LAMBDA_SURPRISE, surprise_weight=1.0,
                     t_max=TRUST_MAX, eta=ETA) if submitted_v1 else
                dict(loss_aversion=AGENT_LOSS_AVERSION, exposed_betrayal_weight=EXPOSED_BETRAYAL_WEIGHT,
                     attribution=ATTRIBUTION, confidence_scoring=CONFIDENCE_SCORING,
                     lambda_surprise=AGENT_LAMBDA_SURPRISE, surprise_weight=None,
                     t_max=AGENT_T_MAX, eta=AGENT_ETA))
        loss_aversion = spec["loss_aversion"] if loss_aversion is None else loss_aversion
        exposed_betrayal_weight = (spec["exposed_betrayal_weight"] if exposed_betrayal_weight is None
                                   else exposed_betrayal_weight)
        attribution = spec["attribution"] if attribution is None else attribution
        confidence_scoring = spec["confidence_scoring"] if confidence_scoring is None else confidence_scoring
        lambda_surprise = spec["lambda_surprise"] if lambda_surprise is None else lambda_surprise
        surprise_weight = spec["surprise_weight"] if surprise_weight is None else surprise_weight
        t_max = spec["t_max"] if t_max is None else t_max
        eta = spec["eta"] if eta is None else eta
        surprise_scale = SURPRISE_SCALE if surprise_scale is None else surprise_scale

        super().__init__(stochastic=stochastic, inv_temp=inv_temp, payoff_matrix=payoff_matrix)

        if surprise_weight is None:
            u_game = stag_indifference_threshold(self.payoff_matrix)
            surprise_weight = round(surprise_scale * (1.0 - u_game) / u_game, 10)
        
        # Validations
        if not 0 <= memory_discount <= 1:
            raise ValueError("memory_discount must be in [0,1]")
        if not 0 <= trust_discount <= 1:
            raise ValueError("trust_discount must be in [0,1]")
        if not 0 <= trust_smoothing <= 1:
            raise ValueError("trust_smoothing must be in [0,1]")
        if u_i is not None and not 0 <= u_i <= 1:
            raise ValueError("u_i must be between 0 and 1")
        if not 0 <= eta <= 1:
            raise ValueError("eta must be in [0,1]")
        if noise_sigma < 0:
            raise ValueError("noise_sigma must be non-negative")
        if loss_aversion < 1:
            raise ValueError("loss_aversion must be at least 1")
        if not 0 <= lambda_surprise <= loss_aversion:
            raise ValueError("lambda_surprise must be in [0, loss_aversion]")
        if exposed_betrayal_weight < 1:
            raise ValueError("exposed_betrayal_weight (nu) must be at least 1")
        if confidence_scoring not in ("hit", "graded"):
            raise ValueError("confidence_scoring must be 'hit' or 'graded'")
        if surprise_weight <= 0:
            raise ValueError("surprise_weight (psi) must be positive")
        if not t_min < t_max:
            raise ValueError("t_min must be less than t_max")
        if eps <= 0:
            raise ValueError("eps must be positive")
        
        # State variables
        self.x = u_i if u_i is not None else 1 - stag_indifference_threshold(self.payoff_matrix)
        self.t = t_init
        
        # Parameters
        self.eta = eta
        self.noise_sigma = noise_sigma
        self.memory_discount = memory_discount
        self.trust_discount = trust_discount
        self.trust_smoothing = trust_smoothing
        self.loss_aversion = loss_aversion
        self.lambda_surprise = lambda_surprise
        self.t_min = t_min
        self.t_max = t_max
        self.eps = eps
        self.exposed_betrayal_weight = float(exposed_betrayal_weight)
        self.surprise_weight = float(surprise_weight)
        self.attribution = bool(attribution)
        self.confidence_scoring = confidence_scoring
        self.submitted_v1 = bool(submitted_v1)
        self.own_action_history = []
        self._last_action = None
        self._threshold = stag_indifference_threshold(self.payoff_matrix)
        self._graded_num = 0.0
        self._graded_den = 0.0
        
        # History tracking
        self.trust_match_hist = []
        self.trust_violation_hist = []
        self.trust_surprise_hist = []
        self.action_history = []
        
        # For compatibility
        self.history_x = [self.x]
        self.history_t = [self.t]
    
    def expected_p(self) -> float:
        """
        Expected cooperation probability from dual-state belief.
        
        Uses Beta distribution parameterization:
        α = eps + x * t
        β = eps + (1 - x) * t
        E[p] = α / (α + β)
        
        Returns:
            float: Expected probability of partner cooperation
        """
        alpha = self.eps + self.x * self.t
        beta = self.eps + (1 - self.x) * self.t
        return alpha / (alpha + beta)
    
    def decide(self) -> int:
        action = super().decide()
        self._last_action = action
        return action

    def update(self, partner_choice: int):
        """
        Update both trust and signal based on observed partner choice.
        
        Args:
            partner_choice: Partner's action (1=cooperate, 0=defect)
        """
        own = self._last_action if self._last_action is not None else int(self.expected_p() >= self._threshold)
        self.own_action_history.append(own)
        self._last_action = None
        new_trust = self._compute_new_trust(partner_choice)
        new_signal = self._compute_new_signal(partner_choice)
        self.t = new_trust
        self.x = new_signal
        
        # Record history
        self.history_x.append(self.x)
        self.history_t.append(self.t)
    
    def _compute_new_trust(self, partner_choice: int) -> float:
        """
        Asymmetric trust update with loss aversion.
        
        Categories (prediction outcomes, independent of the agent's own action):
        - match (expected == observed)             → +1 to numerator & denominator
        - expectation violation (expected C, saw D) → +λ to denominator (penalty)
        - surprise (expected D, saw C)             → +μ to denominator (smaller penalty)
        
        Args:
            partner_choice: Partner's action (1=cooperate, 0=defect)
            
        Returns:
            float: Updated trust level
        """
        expected = 1 if self.x >= 0.5 else 0
        obs = int(partner_choice)

        if self.confidence_scoring == "graded":
            return self._compute_new_trust_graded(expected, obs)
        
        is_match = 1.0 if (obs == expected) else 0.0
        is_violation = 1.0 if (expected == 1 and obs == 0) else 0.0
        is_surprise = 1.0 if (expected == 0 and obs == 1) else 0.0
        
        # Include-current sequences
        M = (self.trust_match_hist + [is_match]) if self.trust_match_hist else [is_match]
        B = (self.trust_violation_hist + [is_violation]) if self.trust_violation_hist else [is_violation]
        U = (self.trust_surprise_hist + [is_surprise]) if self.trust_surprise_hist else [is_surprise]
        
        # Apply exponential discounting
        nM = len(M)
        wM = [self.trust_discount ** (nM - 1 - i) for i in range(nM)]
        nB = len(B)
        wB = [self.trust_discount ** (nB - 1 - i) for i in range(nB)]
        nU = len(U)
        wU = [self.trust_discount ** (nU - 1 - i) for i in range(nU)]
        
        WM = sum(e * w for e, w in zip(M, wM))
        WB = sum(e * w for e, w in zip(B, wB))
        WU = sum(e * w for e, w in zip(U, wU))
        
        # Asymmetric consistency measure
        consistency = WM / (WM + self.loss_aversion * WB + self.lambda_surprise * WU + 1e-9)
        
        target_trust = self.t_min + (self.t_max - self.t_min) * consistency
        new_trust = (1 - self.trust_smoothing) * self.t + self.trust_smoothing * target_trust
        new_trust = float(np.clip(new_trust, self.t_min, self.t_max))
        
        # Record for next iteration
        self.trust_match_hist.append(is_match)
        self.trust_violation_hist.append(is_violation)
        self.trust_surprise_hist.append(is_surprise)
        
        return new_trust
    
    def _compute_new_signal(self, partner_choice: int) -> float:
        """
        Recency-weighted EWMA update with noise.
        
        Args:
            partner_choice: Partner's action (1=cooperate, 0=defect)
            
        Returns:
            float: Updated signal value
        """
        obs = int(partner_choice)
        
        if (self.exposed_betrayal_weight != 1.0 or self.surprise_weight != 1.0) and self.action_history:
            actions = self.action_history + [obs]
            n = len(actions)
            own = self.own_action_history[-n:]
            nu = self.exposed_betrayal_weight
            psi = self.surprise_weight
            w = [(self.memory_discount ** (n - 1 - i))
                 * (nu if (own[i] == 1 and actions[i] == 0
                           and not (self.attribution and i > 0 and own[i - 1] == 0))
                    else psi if (own[i] == 0 and actions[i] == 1) else 1.0)
                 for i in range(n)]
            tot = sum(w)
            P_obs = (sum(a * wi for a, wi in zip(actions, w)) / tot) if tot > 0 else float(np.mean(actions))
        elif self.action_history:
            actions = self.action_history + [obs]
            n = len(actions)
            weights = [self.memory_discount ** (n - 1 - i) for i in range(n)]
            P_obs = sum(a * w for a, w in zip(actions, weights)) / sum(weights)
        else:
            P_obs = float(obs)
        
        noise = np.random.normal(0, self.noise_sigma)
        new_signal = float(np.clip(self.x + self.eta * (P_obs - self.x) + noise, 0.0, 1.0))
        
        self.action_history.append(obs)
        return new_signal

    def _compute_new_trust_graded(self, expected: int, obs: int) -> float:
        """
        consistency = sum_j g^j s_j / sum_j g^j (s_j + w_j (1 - s_j)),   g = trust_discount

        s = probability the belief x gave to what happened (x if the partner collaborated, 1 - x if not), and
        w = 1 for a match, lambda for a violation (expected C, saw D), mu for a surprise (expected D, saw C).
        The three categories of principle 3 are the weights; the miss mass 1 - s carries them. For x in {0, 1}
        this is WM / (WM + lambda WB + mu WU), the submitted measure.
        """
        if obs == expected:
            w = 1.0
        else:
            w = self.loss_aversion if expected == 1 else self.lambda_surprise
        s = self.x if obs == 1 else 1.0 - self.x
        g = self.trust_discount
        self._graded_num = g * self._graded_num + s
        self._graded_den = g * self._graded_den + s + w * (1.0 - s)
        consistency = self._graded_num / (self._graded_den + 1e-9)

        target_trust = self.t_min + (self.t_max - self.t_min) * consistency
        new_trust = (1 - self.trust_smoothing) * self.t + self.trust_smoothing * target_trust
        new_trust = float(np.clip(new_trust, self.t_min, self.t_max))

        # same bookkeeping as the hit rule, so anything reading the categorization histories still works
        self.trust_match_hist.append(1.0 if obs == expected else 0.0)
        self.trust_violation_hist.append(1.0 if (expected == 1 and obs == 0) else 0.0)
        self.trust_surprise_hist.append(1.0 if (expected == 0 and obs == 1) else 0.0)
        return new_trust
