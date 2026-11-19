"""
Co-adaptive partners: learners that pursue their own payoff in the same game as the focal agent.

Exploratory: these partners are not part of the results reported in the paper. Unlike the scripted
(fixed or reactive) and exploitative partners, each has a payoff stake of its own, learns from the
agent, and can settle into mutual collaboration by itself.

Partners and agents see the same payoffs and play the same game, so every class checks that the matrix
is symmetric and refuses an asymmetric one.

  MyopicBestResponsePartner  Stern, Valencia-Romero & Grogan (2022), Design Science 8, doi
                             10.1017/dsj.2022.2, after Kandori, Mailath & Rob (1993) and Blume.
                             Best-responds to the agent's last action, switching with probability
                             0.9 and otherwise repeating itself. In a 2x2 Stag Hunt the best
                             response to the last action is to copy it (R > M and P > S), so in this
                             dyad the partner is Tit-for-Tat with inertia: a replication anchor for
                             that group's model and a control, not a discriminating test.
  EWAPartner                 Camerer & Ho (1999), Econometrica 67:827-874, doi
                             10.1111/1468-0262.00054. Experience-weighted attraction: the experience
                             weight (2.1), the attraction update (2.2) and the logit rule (2.11).
                             Nests choice reinforcement (delta = 0, rho = 0, N(0) = 1) and weighted
                             fictitious play (delta = 1, phi = rho).
  AspirationPartner          Chasparis, Arapostathis & Shamma (2013), SIAM J. Control Optim.
                             51:465-490, the algorithm of their Table 3.1. Keeps its action while
                             the payoff meets a fading-memory aspiration and experiments when it
                             does not. A satisficer, not an optimizer: it holds any action whose
                             payoff clears the aspiration, which in this game includes answering a
                             collaborator with independence (M = 3 clears a prior of 2.25).
  LenientPartner             Lenient Frequency Adjusted Q-learning. Leniency from Panait, Tuyls &
                             Luke (2008), JMLR 9:423-457; the LFAQ form and its Stag Hunt use from
                             Bloembergen, Kaisers & Tuyls (AAMAS 2011) and Bloembergen, de Jong &
                             Tuyls (BNAIC 2011, 3-player Stag Hunt). Updates an action only from the
                             best of its last kappa payoffs, so early miscoordination is forgiven.
  AdaptivePlayPartner        Young (1993), Econometrica 61:57-84. Best-responds to a random sample
                             of k of the agent's last m actions, breaking ties at random, with a
                             small mistake rate. Registry name 'AdaptivePlay', not to be confused
                             with the retired belief-driven 'Adaptive' partner.

Interface, as for every other partner: decide(round_num, last_agent_choice=None), observe() and
update(agent_action), which the runner calls after decide(). A learner needs its own last action to
score the round, so decide() records it. Randomness goes through the `random` module, which
run_agent_simulation seeds per run from the simulation's RNG; that is the pattern the other partners
use, so these partners need no set_rng().

None of them defines coop_prob(): their collaboration probability is endogenous, so the metric panel
treats them as unscheduled and reads p*_k from the realized actions.
"""

import math
import random

import numpy as np

from ..config import as_payoff_matrix

# ---------------------------------------------------------------------------
# Defaults. They live here rather than in registry.py (which holds the other partners' constants)
# because registry.py imports this module, so the dependency can only run one way.
# ---------------------------------------------------------------------------

# Myopic best response: the switching probability of Stern, Valencia-Romero & Grogan (2022)
MBR_SWITCH_PROB = 0.9

# EWA: Camerer & Ho (1999), Table V, median-effort (coordination) games, 1-segment EWA column.
# Those games are the closest published match to a Stag Hunt: a coordination game with multiple
# Pareto-ranked equilibria, estimated over 10-round sessions.
EWA_PHI = 0.800            # decay of past attractions
EWA_RHO = 0.000            # decay of the experience weight (0: attractions cumulate)
EWA_DELTA = 0.853          # weight on the foregone payoff of the action not taken
EWA_N0 = 0.647             # initial experience weight
EWA_LAMBDA = 6.827         # payoff sensitivity of the logit rule
# Payoff range of the game those estimates come from (their Table IV runs from -1.10 to 1.30).
# lambda has units of 1 / payoff, so EWAPartner rescales it by the range of the game actually
# played: 6.827 * 2.40 / 4.00 = 4.10 at the study matrix, and it tracks u across the sweep. Passing
# lam= explicitly overrides the rescaling.
EWA_LAMBDA_PAYOFF_RANGE = 2.40
EWA_A0 = (0.0, 0.0)        # ours: neutral, so round 1 is a fair coin

# Aspiration learning. c and h are the paper's worked examples; the rates are ours, horizon-matched:
# the paper's runs are 10^4 to 10^5 steps (step size 1e-3) and ours are 100 rounds, so its step size
# would never move here. ASPIRATION_STEP = 0.1 gives the same ~10-round memory as the agents'
# gamma_x = 0.9. The default aspiration is the mean payoff, which is the expected payoff of mutual
# random play (2.25 at the study matrix).
ASPIRATION_STEP = 0.1           # ours: step size of the fading-memory aspiration
ASPIRATION_C = 0.2              # paper: slope of the switch probability in the shortfall
ASPIRATION_H = 0.01             # paper: floor on the stay probability
ASPIRATION_TREMBLE_PROB = 0.05  # ours: chance of a perturbation to the aspiration
ASPIRATION_TREMBLE_SIZE = 0.1   # ours: half-width of that perturbation

# Lenient FAQ. kappa is the paper's value; alpha, beta and tau are ours, horizon-matched for the
# same reason (the lenient papers run 5 x 10^4 steps at alpha = 1e-5).
LENIENT_KAPPA = 3     # paper: payoffs collected per action before its value is updated from the best
LENIENT_ALPHA = 0.3   # ours: step size
LENIENT_BETA = 0.5    # ours: frequency adjustment
LENIENT_TAU = 0.1     # ours: Boltzmann temperature
LENIENT_Q0 = 0.5      # ours: neutral, midpoint of the [0, 1] reward scale

# Adaptive play. Young gives the structure, not values, so all three are ours. m = 10 matches the
# window benchmark and the agents' 10-round memory, and k / m = 1/2 meets the usual sampling bound.
ADAPTIVE_PLAY_MEMORY = 10
ADAPTIVE_PLAY_SAMPLE = 5
ADAPTIVE_PLAY_MISTAKE = 0.05


def require_symmetric_game(payoff_matrix) -> np.ndarray:
    """
    Validate and return a symmetric payoff matrix.

    These partners read their own payoffs out of the agent's matrix, and the study is restricted to
    partners and agents who see the same payoffs and play the same game.
    Symmetry means the partner's payoff for (agent a, partner p) equals the agent's payoff for
    (agent p, partner a), i.e. m[a, p, 1] == m[p, a, 0] for every pair of actions.
    """
    m = as_payoff_matrix(payoff_matrix)
    if not np.array_equal(m[:, :, 1], m[:, :, 0].T):
        raise ValueError(
            "co-adaptive partners require a symmetric game (m[a, p, 1] == m[p, a, 0]); the "
            "asymmetric game is out of scope for this study"
        )
    return m


def _ties(a: float, b: float, scale: float) -> bool:
    """
    Whether two expected payoffs should count as equal.

    Not an exact comparison: at the knife-edge where the sampled frequency equals u the two
    expectations are equal in exact arithmetic but differ by an ulp or two once they are summed in
    a different order, which would hand a genuine tie to floating-point noise instead of the coin
    the rule calls for. The sweep hits that knife-edge whenever the sample size divides u.
    """
    return abs(a - b) <= 1e-9 * max(1.0, scale)


def _logistic(z: float) -> float:
    """1 / (1 + exp(-z)), written so neither branch overflows (attractions can grow large)."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class CoAdaptivePartner:
    """
    Shared bookkeeping for the learner partners: the game, own payoffs, and the two histories.

    Subclasses implement decide() and extend update(). The runner calls decide() then update(), so
    decide() records the action it returns in self.last_action and update() scores the round with it.
    """

    def __init__(self, payoff_matrix=None, first_action=None):
        self.payoffs = require_symmetric_game(payoff_matrix)
        self.name = self.__class__.__name__
        self.first_action = None if first_action is None else int(first_action)
        self.history = []        # the agent's past actions, as BasePartner keeps them
        self.own_history = []    # this partner's own past actions
        self.last_action = None

    # -- the game ----------------------------------------------------------
    def own_payoff(self, own_action: int, agent_action: int) -> float:
        """This partner's payoff. The game is symmetric, so it reads its own row of the matrix."""
        return float(self.payoffs[int(own_action), int(agent_action), 0])

    def expected_own_payoff(self, own_action: int, prob_agent_stag: float) -> float:
        """Expected payoff of own_action when the agent collaborates with probability q."""
        q = float(prob_agent_stag)
        return (1.0 - q) * self.own_payoff(own_action, 0) + q * self.own_payoff(own_action, 1)

    @property
    def payoff_span(self) -> float:
        """Range of this partner's payoffs in the game, used to put parameters on its scale."""
        own = self.payoffs[:, :, 0]
        return float(own.max() - own.min())

    # -- interface ---------------------------------------------------------
    def opening_action(self) -> int:
        """Round-1 action: first_action if one was set, otherwise a fair coin (no built-in stance)."""
        if self.first_action is not None:
            return self.first_action
        return 1 if random.random() < 0.5 else 0

    def record(self, action: int) -> int:
        """Record and return the action decide() is about to hand to the runner."""
        self.last_action = int(action)
        return self.last_action

    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        raise NotImplementedError("subclasses must implement decide()")

    def observe(self, focal_agent_action: int):
        pass

    def update(self, agent_action: int):
        """Subclasses extend this; it keeps both histories."""
        self.history.append(int(agent_action))
        self.own_history.append(self.last_action)

    def reset(self):
        self.history = []
        self.own_history = []
        self.last_action = None


class MyopicBestResponsePartner(CoAdaptivePartner):
    """
    Best-responds to the agent's last action, switching with probability `switch_prob`.

    Stern, Valencia-Romero & Grogan (2022). With probability switch_prob the partner moves to the
    action that would have maximized its payoff last round; otherwise it repeats itself (inertia).
    """

    def __init__(self, payoff_matrix=None, switch_prob=MBR_SWITCH_PROB, first_action=None):
        super().__init__(payoff_matrix, first_action)
        if not 0.0 <= switch_prob <= 1.0:
            raise ValueError("switch_prob must be in [0, 1]")
        self.switch_prob = float(switch_prob)

    def best_response(self, agent_action: int):
        """
        Best reply to agent_action, or None if the two actions tie (never in a Stag Hunt).

        These are matrix entries, not sums, so an exact comparison is safe here; the sampled rule in
        AdaptivePlayPartner needs a tolerance because it averages first.
        """
        hare, stag = (self.own_payoff(p, agent_action) for p in (0, 1))
        if stag == hare:
            return None
        return 1 if stag > hare else 0

    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        if last_agent_choice is None or self.last_action is None:
            return self.record(self.opening_action())
        br = self.best_response(last_agent_choice)
        if br is None or br == self.last_action:
            return self.record(self.last_action)        # already there, or indifferent: no draw
        if self.switch_prob >= 1.0:
            # The deterministic limit has to reproduce the matching reactive partner bit for bit, so
            # it must not consume a draw the reactive partner does not make.
            return self.record(br)
        return self.record(br if random.random() < self.switch_prob else self.last_action)


class EWAPartner(CoAdaptivePartner):
    """
    Experience-weighted attraction learning (Camerer & Ho 1999).

    Each action carries an attraction, updated from the payoff it earned or (weighted by delta) the
    payoff it would have earned, normalized by an experience weight:

        N(t)    = rho N(t-1) + 1                                                         Eq. (2.1)
        A_j(t)  = [phi N(t-1) A_j(t-1) + (delta + (1 - delta) I(j chosen)) pi_j(t)] / N(t)   (2.2)
        P(stag) = logit over lambda A                                                        (2.11)

    The action is drawn from the attractions the previous round left, which is Eq. (2.11)'s timing.
    """

    def __init__(self, payoff_matrix=None, phi=EWA_PHI, rho=EWA_RHO, delta=EWA_DELTA,
                 n0=EWA_N0, lam=None, a0=EWA_A0, first_action=None):
        super().__init__(payoff_matrix, first_action)
        self.phi, self.rho, self.delta = float(phi), float(rho), float(delta)
        self.n0 = float(n0)
        self.a0 = (float(a0[0]), float(a0[1]))
        # lambda has units of 1 / payoff, so the published estimate is rescaled to this game unless
        # an explicit value is given (the sensitivity grid passes one).
        self.lam = (EWA_LAMBDA * EWA_LAMBDA_PAYOFF_RANGE / self.payoff_span
                    if lam is None else float(lam))
        self.A = list(self.a0)
        self.N = self.n0

    def prob_stag(self) -> float:
        """Logit choice rule, written on the attraction gap so it cannot overflow."""
        return _logistic(self.lam * (self.A[1] - self.A[0]))

    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        if self.first_action is not None and self.last_action is None:
            return self.record(self.first_action)
        return self.record(1 if random.random() < self.prob_stag() else 0)

    def update(self, agent_action: int):
        chosen = self.last_action
        n_prev = self.N
        self.N = self.rho * n_prev + 1.0
        self.A = [
            (self.phi * n_prev * self.A[j]
             + (1.0 if j == chosen else self.delta) * self.own_payoff(j, agent_action)) / self.N
            for j in (0, 1)
        ]
        super().update(agent_action)

    def reset(self):
        super().reset()
        self.A = list(self.a0)
        self.N = self.n0


class AspirationPartner(CoAdaptivePartner):
    """
    Aspiration learning (Chasparis, Arapostathis & Shamma 2013, Table 3.1).

    The partner holds an aspiration level, a perturbed fading-memory average of its payoffs. After
    each round it compares the payoff it got with the aspiration it held:

        z          = u(t) - rho(t)
        rho(t+1)   = sat[rho(t) + step z + tremble]
        stay prob  = 1 if z >= 0, else max(h, 1 + c z)

    A content player repeats its action, a discontent one switches with the complementary
    probability (with two actions there is only one other action to move to). The aspiration is
    clipped to a band that strictly contains the game's payoffs, as the paper requires.
    """

    def __init__(self, payoff_matrix=None, step=ASPIRATION_STEP, c=ASPIRATION_C, h=ASPIRATION_H,
                 tremble_prob=ASPIRATION_TREMBLE_PROB, tremble_size=ASPIRATION_TREMBLE_SIZE,
                 aspiration0=None, first_action=None):
        super().__init__(payoff_matrix, first_action)
        if not 0.0 < h < 1.0:
            raise ValueError("h must be in (0, 1)")
        self.step, self.c, self.h = float(step), float(c), float(h)
        self.tremble_prob, self.tremble_size = float(tremble_prob), float(tremble_size)
        own = self.payoffs[:, :, 0]
        self.bounds = (float(own.min()) - 1.0, float(own.max()) + 1.0)
        # Default: the mean payoff, which is the expected payoff of mutual random play.
        self.aspiration0 = float(own.mean()) if aspiration0 is None else float(aspiration0)
        self.aspiration = self.aspiration0
        self.next_action = None

    def stay_prob(self, z: float) -> float:
        """phi(z) of Table 3.1: content repeats for sure, discontent repeats less the worse z is."""
        if z >= 0.0:
            return 1.0
        return max(self.h, 1.0 + self.c * z)

    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        if self.next_action is None:
            return self.record(self.opening_action())
        return self.record(self.next_action)

    def update(self, agent_action: int):
        # Both steps of Table 3.1 read the aspiration the partner held while it played, so z first.
        z = self.own_payoff(self.last_action, agent_action) - self.aspiration
        tremble = 0.0
        if self.tremble_prob > 0.0 and random.random() < self.tremble_prob:
            tremble = random.uniform(-self.tremble_size, self.tremble_size)
        lo, hi = self.bounds
        self.aspiration = min(hi, max(lo, self.aspiration + self.step * z + tremble))
        stay = self.stay_prob(z)
        if stay >= 1.0 or random.random() < stay:   # content: no draw, and it never switches
            self.next_action = self.last_action
        else:
            self.next_action = 1 - self.last_action
        super().update(agent_action)

    def reset(self):
        super().reset()
        self.aspiration = self.aspiration0
        self.next_action = None


class LenientPartner(CoAdaptivePartner):
    """
    Lenient Frequency Adjusted Q-learning (Panait, Tuyls & Luke 2008; Bloembergen et al. 2011).

    The partner collects kappa payoffs for an action before updating that action's value, and then
    updates from the BEST of them, so early miscoordination is forgiven:

        Q_i <- Q_i + min(beta / x_i, 1) alpha (max of the kappa payoffs - Q_i)
        x   =  Boltzmann over Q / tau

    The frequency adjustment min(beta / x_i, 1) equalizes the expected learning speed of an action
    that is chosen often and one that is not. Payoffs are mapped to [0, 1], the range the lenient
    model is defined on. kappa = 1 is plain Frequency Adjusted Q-learning.
    """

    def __init__(self, payoff_matrix=None, kappa=LENIENT_KAPPA, alpha=LENIENT_ALPHA,
                 beta=LENIENT_BETA, tau=LENIENT_TAU, q0=LENIENT_Q0, first_action=None):
        super().__init__(payoff_matrix, first_action)
        if int(kappa) < 1:
            raise ValueError("kappa must be at least 1")
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        self.kappa = int(kappa)
        self.alpha, self.beta, self.tau = float(alpha), float(beta), float(tau)
        self.q0 = float(q0)
        own = self.payoffs[:, :, 0]
        self._reward_lo = float(own.min())
        self.Q = [self.q0, self.q0]
        self.buffers = ([], [])

    def scaled_reward(self, own_action: int, agent_action: int) -> float:
        """The round's payoff on the [0, 1] scale the lenient model assumes."""
        return (self.own_payoff(own_action, agent_action) - self._reward_lo) / self.payoff_span

    def prob_stag(self) -> float:
        """Boltzmann exploration; with two actions it is a logistic on the Q gap over tau."""
        return _logistic((self.Q[1] - self.Q[0]) / self.tau)

    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        if self.first_action is not None and self.last_action is None:
            return self.record(self.first_action)
        return self.record(1 if random.random() < self.prob_stag() else 0)

    def update(self, agent_action: int):
        i = self.last_action
        self.buffers[i].append(self.scaled_reward(i, agent_action))
        if len(self.buffers[i]) >= self.kappa:
            reward = max(self.buffers[i])           # leniency: forgive all but the best of the kappa
            self.buffers[i].clear()
            x_i = self.prob_stag() if i == 1 else 1.0 - self.prob_stag()
            rate = self.alpha * (min(self.beta / x_i, 1.0) if x_i > 0.0 else 1.0)
            self.Q[i] += rate * (reward - self.Q[i])
        super().update(agent_action)

    def reset(self):
        super().reset()
        self.Q = [self.q0, self.q0]
        self.buffers = ([], [])


class AdaptivePlayPartner(CoAdaptivePartner):
    """
    Adaptive play (Young 1993).

    Each round the partner draws a sample of `sample` of the agent's last `memory` actions, best
    responds to the frequency of collaboration in that sample, breaks ties at random, and with
    probability `mistake_prob` plays at random instead. In 2x2 coordination games the perturbed
    process selects the risk-dominant convention, which at u = 2/3 is mutual independence.

    The sample is drawn without replacement, and is the whole window while fewer than `sample`
    rounds have been seen (then no draw is needed, which makes sample >= memory deterministic).
    """

    def __init__(self, payoff_matrix=None, memory=ADAPTIVE_PLAY_MEMORY, sample=ADAPTIVE_PLAY_SAMPLE,
                 mistake_prob=ADAPTIVE_PLAY_MISTAKE, first_action=None):
        super().__init__(payoff_matrix, first_action)
        if int(memory) < 1 or int(sample) < 1:
            raise ValueError("memory and sample must be at least 1")
        if not 0.0 <= mistake_prob <= 1.0:
            raise ValueError("mistake_prob must be in [0, 1]")
        self.memory, self.sample = int(memory), int(sample)
        self.mistake_prob = float(mistake_prob)

    def sampled_best_reply(self) -> int:
        """Best reply to the collaboration frequency of a random sample of the recent window."""
        window = self.history[-self.memory:]
        draw = window if self.sample >= len(window) else random.sample(window, self.sample)
        q = sum(draw) / len(draw)
        hare, stag = (self.expected_own_payoff(p, q) for p in (0, 1))
        if _ties(stag, hare, self.payoff_span):
            return 1 if random.random() < 0.5 else 0
        return 1 if stag > hare else 0

    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        if self.mistake_prob > 0.0 and random.random() < self.mistake_prob:
            return self.record(1 if random.random() < 0.5 else 0)
        if not self.history:
            return self.record(self.opening_action())
        return self.record(self.sampled_best_reply())
