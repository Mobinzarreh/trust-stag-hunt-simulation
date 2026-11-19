"""
Held-out partners: an exploratory validation set, not part of the results reported in the paper.

They were specified after the revised agent was fixed and are kept out of every other partner tuple
(registry.HELDOUT_PARTNERS), so no study script picks them up.

    GenerousTFT    reciprocator that keeps offering collaboration from a stalemate (re-engagement test)
    Reformer       starts unreliable, becomes reliable: trust built from distrust with no exposed betrayal
    Breach         reliable with one short block of independence early, late or never (timing of a breach)
    RegimeSwitch   Markov switching between a cooperative and an exploitative regime (change nobody scheduled)

Every schedule is a fraction of num_rounds, so the horizon sweep can stretch it. The scheduled partners define
coop_prob(round) for the metric panel. RegimeSwitch deliberately does not: its p*_k differs from run to run, and the
panel builds p* by tiling one fresh instance, so it logs its own p*_k in pstar_log instead.
"""

import random


class GenerousTitForTatPartner:
    """
    Start with collaboration; afterwards copy the agent's last action, except that after an agent independence choice
    it still collaborates with probability `generosity`.

    The default g = 1/3 is the Prisoner's Dilemma value of Nowak & Sigmund (1992), g = min(1 - (T - R)/(R - S),
    (R - P)/(T - P)) at T, R, P, S = 5, 3, 1, 0. That formula has no stag-hunt counterpart, so here it is a borrowed
    constant. Reactive: no coop_prob, so the schedule-based panel metrics are empty, as for TitForTat.
    """

    def __init__(self, generosity: float = 1 / 3):
        if not 0 <= generosity <= 1:
            raise ValueError("generosity must be in [0, 1]")
        self.generosity = generosity

    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        if last_agent_choice is None or int(last_agent_choice) == 1:
            return 1
        return 1 if random.random() < self.generosity else 0

    def observe(self, focal_agent_action: int):
        pass

    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


class _ScheduledPartner:
    """A partner that collaborates with probability coop_prob(round) in each round, independently of the agent."""

    def coop_prob(self, round_num: int) -> float:
        raise NotImplementedError

    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        return 1 if random.random() < self.coop_prob(round_num) else 0

    def observe(self, focal_agent_action: int):
        pass

    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


class ReformerPartner(_ScheduledPartner):
    """
    Unreliable (p_low) until reform_fraction * num_rounds, reliable (p_high) from then on.

    Built to test trust built from initial distrust (ignorance, not betrayal): the early independence choices mostly
    meet an agent that is holding back, so they are rarely exposed betrayals.
    """

    def __init__(self, num_rounds: int = 100, reform_fraction: float = 0.4,
                 p_low: float = 0.15, p_high: float = 0.9):
        if not 0 <= p_low <= p_high <= 1:
            raise ValueError("need 0 <= p_low <= p_high <= 1")
        if not 0 < reform_fraction < 1:
            raise ValueError("need 0 < reform_fraction < 1")
        self.reform_round = int(round(num_rounds * reform_fraction))
        self.p_low = p_low
        self.p_high = p_high

    def coop_prob(self, round_num: int) -> float:
        return self.p_low if round_num < self.reform_round else self.p_high


class BreachPartner(_ScheduledPartner):
    """
    Reliable (p_high) except for one block of rounds at p_breach, starting at breach_fraction * num_rounds.

    breach_fraction=None gives the no-breach control. The block lasts max(1, round(block_fraction * num_rounds))
    rounds (3 at K = 100).
    """

    def __init__(self, num_rounds: int = 100, breach_fraction=None, block_fraction: float = 0.03,
                 p_high: float = 0.95, p_breach: float = 0.0):
        if not 0 <= p_breach <= p_high <= 1:
            raise ValueError("need 0 <= p_breach <= p_high <= 1")
        self.p_high = p_high
        self.p_breach = p_breach
        if breach_fraction is None:
            self.breach_start = self.breach_end = None
        else:
            length = max(1, int(round(block_fraction * num_rounds)))
            self.breach_start = int(round(breach_fraction * num_rounds))
            self.breach_end = self.breach_start + length
            if self.breach_end > num_rounds:
                raise ValueError("the breach block must end inside the interaction")

    def coop_prob(self, round_num: int) -> float:
        if self.breach_start is not None and self.breach_start <= round_num < self.breach_end:
            return self.p_breach
        return self.p_high


class RegimeSwitchPartner:
    """
    Two hidden regimes: cooperative (p_coop) and exploitative (p_exploit). Starts cooperative; after its action in
    each round it switches regime with probability `hazard`.

    The switch draw comes after the action draw, from the same Python `random` stream the runner seeds per run, so a
    run is reproducible. p*_k (the collaboration probability actually used in round k) is appended to pstar_log.
    There is no coop_prob(): the panel would tile one instance's path across all runs, which is wrong here.
    """

    def __init__(self, p_coop: float = 0.9, p_exploit: float = 0.2, hazard: float = 0.05, start_coop: bool = True):
        if not (0 <= p_exploit <= 1 and 0 <= p_coop <= 1 and 0 <= hazard <= 1):
            raise ValueError("probabilities must be in [0, 1]")
        self.p_coop = p_coop
        self.p_exploit = p_exploit
        self.hazard = hazard
        self.coop_regime = bool(start_coop)
        self.pstar_log = []

    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        p = self.p_coop if self.coop_regime else self.p_exploit
        self.pstar_log.append(p)
        action = 1 if random.random() < p else 0
        if random.random() < self.hazard:
            self.coop_regime = not self.coop_regime
        return action

    def observe(self, focal_agent_action: int):
        pass

    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass
