"""Fixed strategy partners."""

import numpy as np
import random

from .base import BasePartner


class AlwaysCooperatePartner:
    """Always cooperates regardless of agent's actions."""
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Always return cooperate."""
        return 1

    def coop_prob(self, round_num: int) -> float:
        return 1.0
    
    def observe(self, focal_agent_action: int):
        pass
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


# Alias for compatibility
AlwaysCollaboratePartner = AlwaysCooperatePartner


class AlwaysDefectPartner:
    """Always defects regardless of agent's actions."""
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Always return defect."""
        return 0

    def coop_prob(self, round_num: int) -> float:
        return 0.0
    
    def observe(self, focal_agent_action: int):
        pass
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


class RandomPartner:
    """Random partner with cooperation probability p."""
    
    def __init__(self, p: float = 0.5):
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0,1]")
        self.p = p
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Randomly choose cooperation or defection."""
        return 1 if random.random() < self.p else 0

    def coop_prob(self, round_num: int) -> float:
        return float(self.p)
    
    def observe(self, focal_agent_action: int):
        pass
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


class PeriodicCheaterPartner:
    """
    Cooperate for `cycle_length` rounds, then defect for `cheat_duration` rounds, repeat.
    
    Creates predictable betrayal cycles to test agent's pattern recognition.
    """
    
    def __init__(self, cycle_length: int = 6, cheat_duration: int = 2):
        """
        Initialize periodic cheater.
        
        Args:
            cycle_length: Number of cooperation rounds in each cycle
            cheat_duration: Number of defection rounds in each cycle
        """
        self.cycle_length = cycle_length
        self.cheat_duration = cheat_duration
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Cooperate for cycle_length rounds, defect for cheat_duration rounds."""
        return 1 if (round_num % (self.cycle_length + self.cheat_duration)) < self.cycle_length else 0

    def coop_prob(self, round_num: int) -> float:
        return float(self.decide(round_num))
    
    def observe(self, focal_agent_action: int):
        pass  # No learning
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


class SingleCyclePartner(BasePartner):
    """
    Cooperate for the first fraction of rounds, then defect thereafter.
    
    Tests agent's response to sudden, permanent strategy change.
    Cooperates for first 30% of total rounds, then permanently defects.
    """
    
    def __init__(self, num_rounds: int = 100, cooperate_fraction: float = 0.3):
        """
        Initialize single cycle partner.
        
        Args:
            num_rounds: Total number of rounds in the simulation
            cooperate_fraction: Fraction of rounds to cooperate before switching (default: 0.3 = 30%)
        """
        super().__init__()
        self.num_rounds = num_rounds
        self.cooperate_fraction = cooperate_fraction
        self.cooperate_rounds = int(num_rounds * cooperate_fraction)
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Cooperate for first cooperate_fraction of rounds, then always defect."""
        return 1 if round_num < self.cooperate_rounds else 0

    def coop_prob(self, round_num: int) -> float:
        return float(self.decide(round_num))


class ProbabilisticPartner:
    """
    Cooperate with a fixed probability.
    """
    
    def __init__(self, cooperate_prob: float = 0.7):
        """
        Initialize probabilistic partner.
        
        Args:
            cooperate_prob: Probability of cooperation (constant)
        """
        if not 0 <= cooperate_prob <= 1:
            raise ValueError("cooperate_prob must be in [0,1]")
        self.cooperate_prob = cooperate_prob
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Cooperate with fixed probability."""
        return 1 if random.random() < self.cooperate_prob else 0

    def coop_prob(self, round_num: int) -> float:
        return float(self.cooperate_prob)

    def observe(self, focal_agent_action: int):
        pass
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


class GradualDeteriorationPartner:
    """
    Starts fully cooperative, then gradually reduces cooperation probability linearly over time.
    cooperation_prob(t) = 1.0 - (deterioration_rate * t / num_rounds)
    """
    
    def __init__(self, deterioration_rate: float = 0.8, num_rounds: int = 100):
        """
        Initialize gradual deterioration partner.
        
        Args:
            deterioration_rate: How much to deteriorate (0.8 = drops to 20% by end)
            num_rounds: Deterioration horizon; set equal to the simulation length
        """
        self.deterioration_rate = deterioration_rate  # how much to deteriorate
        self.num_rounds = num_rounds
    
    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        """Linear deterioration: starts at 1.0, decreases over time."""
        # Linear deterioration: starts at 1.0, decreases over time
        coop_prob = 1.0 - (self.deterioration_rate * round_num / self.num_rounds)
        coop_prob = max(0.0, min(1.0, coop_prob))  # clip to [0,1]
        return 1 if random.random() < coop_prob else 0

    def coop_prob(self, round_num: int) -> float:
        return max(0.0, min(1.0, 1.0 - self.deterioration_rate * round_num / self.num_rounds))
    
    def observe(self, focal_agent_action: int):
        pass
    
    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


class BreakRepairPartner:
    """
    Reliable, then unreliable for a spell, then reliable again (a betrayal episode and its repair).

    Collaborates with probability p_high before break_round and from repair_round on, and with
    probability p_low in between. Scales with num_rounds through the two fractions.
    """

    def __init__(self, num_rounds: int = 100, break_fraction: float = 0.3, repair_fraction: float = 0.6,
                 p_high: float = 0.95, p_low: float = 0.05):
        if not 0 <= p_low <= p_high <= 1:
            raise ValueError("need 0 <= p_low <= p_high <= 1")
        if not 0 < break_fraction < repair_fraction <= 1:
            raise ValueError("need 0 < break_fraction < repair_fraction <= 1")
        self.break_round = int(num_rounds * break_fraction)
        self.repair_round = int(num_rounds * repair_fraction)
        self.p_high = p_high
        self.p_low = p_low

    def coop_prob(self, round_num: int) -> float:
        """True collaboration probability in round round_num."""
        return self.p_low if self.break_round <= round_num < self.repair_round else self.p_high

    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        return 1 if random.random() < self.coop_prob(round_num) else 0

    def observe(self, focal_agent_action: int):
        pass

    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass


class ReliabilityRampPartner:
    """
    Reliability ramps down and back up along the same path (hysteresis probe).

    p_high until down_fraction * num_rounds, linear to p_low at trough_fraction * num_rounds,
    linear back to p_high at up_fraction * num_rounds, then p_high. The two ramps are mirror
    images, so rounds trough - j and trough + j have the same true reliability.
    """

    def __init__(self, num_rounds: int = 100, down_fraction: float = 0.2, trough_fraction: float = 0.5,
                 up_fraction: float = 0.8, p_high: float = 0.95, p_low: float = 0.05):
        if not 0 <= p_low <= p_high <= 1:
            raise ValueError("need 0 <= p_low <= p_high <= 1")
        self.down_start = int(num_rounds * down_fraction)
        self.trough = int(num_rounds * trough_fraction)
        self.up_end = int(num_rounds * up_fraction)
        if not (self.trough - self.down_start == self.up_end - self.trough > 0):
            raise ValueError("the down and up ramps must have the same positive length")
        self.p_high = p_high
        self.p_low = p_low

    def coop_prob(self, round_num: int) -> float:
        """True collaboration probability in round round_num."""
        half = self.trough - self.down_start
        dist = abs(round_num - self.trough)
        if dist >= half:
            return self.p_high
        return self.p_low + (self.p_high - self.p_low) * dist / half

    def decide(self, round_num: int, last_agent_choice: int = None) -> int:
        return 1 if random.random() < self.coop_prob(round_num) else 0

    def observe(self, focal_agent_action: int):
        pass

    def update(self, agent_action: int):
        """Compatibility with simulation runner."""
        pass
