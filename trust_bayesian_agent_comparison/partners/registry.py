"""
Single source of truth for partner configurations.

Every script should build partners through this module rather than constructing
partner classes inline, so that a partner name always means the same behavior
(e.g. the same Single-Cycle switch point or deterioration horizon) across the
main study, sensitivity sweeps, rounds analysis and figures.

Factories returned here are zero-argument, picklable callables (functools.partial
of a module-level function), as expected by MonteCarloManager and joblib.
"""

from functools import partial

from ..config import NUM_ROUNDS, stag_indifference_threshold
from .fixed import (
    AlwaysCooperatePartner,
    AlwaysDefectPartner,
    RandomPartner,
    PeriodicCheaterPartner,
    SingleCyclePartner,
    GradualDeteriorationPartner,
    BreakRepairPartner,
    ReliabilityRampPartner,
    ProbabilisticPartner,
)
from .reactive import (
    TitForTatCooperatePartner,
    SuspiciousTitForTatPartner,
    GrimTriggerPartner,
    PavlovPartner,
)
from .belief_driven import (
    StrategicCheaterPartner,
    DistrustLockPartner,
    ExpectationViolationPartner,
    AdaptivePartner,
)
from .learners import (
    MyopicBestResponsePartner,
    EWAPartner,
    AspirationPartner,
    LenientPartner,
    AdaptivePlayPartner,
)
from .heldout import (
    GenerousTitForTatPartner,
    ReformerPartner,
    BreachPartner,
    RegimeSwitchPartner,
)

# Partners reported in the manuscript (Table 3), in table order
MANUSCRIPT_PARTNERS = (
    'AlwaysCooperate',
    'AlwaysDefect',
    'Random',
    'TitForTat',
    'SuspiciousTFT',
    'SingleCycle',
    'GradualDeterioration',
    'StrategicCheater',
    'ExpectationViolation',
)


RETIRED_PARTNERS = ('PeriodicCheater', 'GrimTrigger', 'Pavlov', 'Adaptive')

ALL_PARTNERS = MANUSCRIPT_PARTNERS

DIAGNOSTIC_PARTNERS = ('BreakRepair', 'ReliabilityRamp', 'DistrustLock')

COLLABORATIVE_PARTNERS = ('MyopicBestResponse', 'EWA', 'AspirationLearner', 'LenientLearner',
                          'AdaptivePlay')

HELDOUT_PARTNERS = ('GenerousTFT', 'Reformer', 'BreachEarly', 'BreachLate', 'BreachNone', 'RegimeSwitch')

PARTNER_CATEGORY = {
    'AlwaysCooperate': 'stationary',
    'AlwaysDefect': 'stationary',
    'Random': 'stationary',
    'PeriodicCheater': 'stationary',
    'TitForTat': 'reactive',
    'SuspiciousTFT': 'reactive',
    'GrimTrigger': 'reactive',
    'Pavlov': 'reactive',
    'SingleCycle': 'non-stationary',
    'GradualDeterioration': 'non-stationary',
    'StrategicCheater': 'non-stationary',
    'ExpectationViolation': 'non-stationary',
    'Adaptive': 'non-stationary',
    'BreakRepair': 'non-stationary',
    'ReliabilityRamp': 'non-stationary',
    'DistrustLock': 'non-stationary',
    'ReliableNoisy': 'stationary',
    'MyopicBestResponse': 'co-adaptive',
    'EWA': 'co-adaptive',
    'AspirationLearner': 'co-adaptive',
    'LenientLearner': 'co-adaptive',
    'AdaptivePlay': 'co-adaptive',
    'GenerousTFT': 'reactive',
    'Reformer': 'non-stationary',
    'BreachEarly': 'non-stationary',
    'BreachLate': 'non-stationary',
    'BreachNone': 'stationary',
    'RegimeSwitch': 'non-stationary',
}

# Fraction of the interaction the Single-Cycle partner collaborates before switching
SINGLE_CYCLE_FRACTION = 0.3
# Total drop in collaboration probability for Gradual Deterioration (1.0 -> 0.2)
DETERIORATION_RATE = 0.8
# Belief-driven Strategic Cheater switches to independence once its estimate of
# the focal agent's confidence exceeds this value (midpoint of [t_min, t_max])
STRATEGIC_CHEATER_THRESHOLD = 5.0
# Deterioration horizon actually used for the submitted manuscript results
# (the class default at the time); only used when submitted_v1=True
SUBMITTED_V1_DETERIORATION_HORIZON = 70
SUBMITTED_V1_LOSS_AVERSION_HAT = 2.0
DIAGNOSTIC_P_HIGH = 0.95
DIAGNOSTIC_P_LOW = 0.05
RELIABLE_NOISY_P = 0.85
BREAK_REPAIR_FRACTIONS = (0.3, 0.6)
RELIABILITY_RAMP_FRACTIONS = (0.2, 0.5, 0.8)
GENEROUS_TFT_G = 1 / 3              # Nowak & Sigmund (1992) PD value, borrowed (no stag-hunt counterpart)
REFORMER_FRACTION = 0.4             # reliable from round 0.4 K on
REFORMER_P = (0.15, 0.9)            # (before, after)
BREACH_FRACTIONS = {'BreachEarly': 0.05, 'BreachLate': 0.5, 'BreachNone': None}
BREACH_BLOCK_FRACTION = 0.03        # 3 rounds at K = 100
BREACH_P_HIGH = 0.95
REGIME_SWITCH = dict(p_coop=0.9, p_exploit=0.2, hazard=0.05)


def build_partner(name: str, num_rounds: int = NUM_ROUNDS, submitted_v1: bool = False,
                  payoff_matrix=None, partner_x0=None):
    """
    Construct a fresh partner instance.

    Args:
        name: Partner name (see ALL_PARTNERS)
        num_rounds: Simulation length; time-indexed partners scale to it
        submitted_v1: If True, reproduce the configuration used for the submitted
            manuscript results. Two differences: the Gradual Deterioration horizon
            (70 rounds instead of num_rounds) and the belief-driven partners' betrayal
            penalty (lambda_hat = 2 instead of the agent-matched LOSS_AVERSION = 3,
            which only moves the Strategic Cheater) and the Strategic Cheater's decision rule (no
            expectation gate, instead of the current x_hat >= u gate).
        payoff_matrix: Game being played (default: config.PAYOFF_MATRIX). Pavlov best-responds
            in it, and belief-driven partners start from x_hat_0 = 1 - u of it, mirroring the
            focal agent's default initial belief.
        partner_x0: Initial belief x_hat_0 of the belief-driven partners, overriding 1 - u
            (used to hold the partners' initialization fixed while the game changes)
    """
    if partner_x0 is not None:
        x0_hat = partner_x0
    elif payoff_matrix is not None:
        x0_hat = 1 - stag_indifference_threshold(payoff_matrix)
    else:
        x0_hat = None
    if name == 'AlwaysCooperate':
        return AlwaysCooperatePartner()
    if name == 'AlwaysDefect':
        return AlwaysDefectPartner()
    if name == 'Random':
        return RandomPartner(p=0.5)
    if name == 'PeriodicCheater':
        return PeriodicCheaterPartner(cycle_length=6, cheat_duration=2)
    if name == 'SingleCycle':
        return SingleCyclePartner(num_rounds=num_rounds, cooperate_fraction=SINGLE_CYCLE_FRACTION)
    if name == 'GradualDeterioration':
        horizon = SUBMITTED_V1_DETERIORATION_HORIZON if submitted_v1 else num_rounds
        return GradualDeteriorationPartner(deterioration_rate=DETERIORATION_RATE, num_rounds=horizon)
    if name == 'TitForTat':
        return TitForTatCooperatePartner()
    if name == 'SuspiciousTFT':
        return SuspiciousTitForTatPartner()
    if name == 'GrimTrigger':
        return GrimTriggerPartner()
    if name == 'Pavlov':
        return PavlovPartner(payoff_matrix=payoff_matrix)
    lam_hat = {'loss_aversion_hat': SUBMITTED_V1_LOSS_AVERSION_HAT} if submitted_v1 else {}
    if name == 'StrategicCheater':
        gate = None if submitted_v1 else stag_indifference_threshold(payoff_matrix)
        return StrategicCheaterPartner(t_threshold=STRATEGIC_CHEATER_THRESHOLD, u_i=x0_hat,
                                       expectation_gate=gate, **lam_hat)
    if name == 'DistrustLock':
        return DistrustLockPartner(t_threshold=STRATEGIC_CHEATER_THRESHOLD, u_i=x0_hat, **lam_hat)
    if name == 'ExpectationViolation':
        return ExpectationViolationPartner(u_i=x0_hat, **lam_hat)
    if name == 'Adaptive':
        return AdaptivePartner(u_i=x0_hat, **lam_hat)
    if name == 'BreakRepair':
        return BreakRepairPartner(num_rounds, *BREAK_REPAIR_FRACTIONS,
                                  p_high=DIAGNOSTIC_P_HIGH, p_low=DIAGNOSTIC_P_LOW)
    if name == 'ReliabilityRamp':
        return ReliabilityRampPartner(num_rounds, *RELIABILITY_RAMP_FRACTIONS,
                                      p_high=DIAGNOSTIC_P_HIGH, p_low=DIAGNOSTIC_P_LOW)
    if name == 'ReliableNoisy':
        return ProbabilisticPartner(cooperate_prob=RELIABLE_NOISY_P)
    if name == 'MyopicBestResponse':
        return MyopicBestResponsePartner(payoff_matrix=payoff_matrix)
    if name == 'EWA':
        return EWAPartner(payoff_matrix=payoff_matrix)
    if name == 'AspirationLearner':
        return AspirationPartner(payoff_matrix=payoff_matrix)
    if name == 'LenientLearner':
        return LenientPartner(payoff_matrix=payoff_matrix)
    if name == 'AdaptivePlay':
        return AdaptivePlayPartner(payoff_matrix=payoff_matrix)
    if name == 'GenerousTFT':
        return GenerousTitForTatPartner(generosity=GENEROUS_TFT_G)
    if name == 'Reformer':
        return ReformerPartner(num_rounds, REFORMER_FRACTION, *REFORMER_P)
    if name in BREACH_FRACTIONS:
        return BreachPartner(num_rounds, BREACH_FRACTIONS[name], BREACH_BLOCK_FRACTION, p_high=BREACH_P_HIGH)
    if name == 'RegimeSwitch':
        return RegimeSwitchPartner(**REGIME_SWITCH)
    raise ValueError(f"Unknown partner '{name}'. Known partners: {tuple(PARTNER_CATEGORY)}")


def partner_factory(name: str, num_rounds: int = NUM_ROUNDS, submitted_v1: bool = False,
                    payoff_matrix=None, partner_x0=None):
    """Picklable zero-argument factory for one partner."""
    if name not in PARTNER_CATEGORY:
        raise ValueError(f"Unknown partner '{name}'. Known partners: {tuple(PARTNER_CATEGORY)}")
    return partial(build_partner, name, num_rounds, submitted_v1, payoff_matrix, partner_x0)


def partner_factories(names=MANUSCRIPT_PARTNERS, num_rounds: int = NUM_ROUNDS,
                      submitted_v1: bool = False, payoff_matrix=None, partner_x0=None) -> dict:
    """Dict of {name: factory} for the given partner names."""
    return {name: partner_factory(name, num_rounds, submitted_v1, payoff_matrix, partner_x0)
            for name in names}
