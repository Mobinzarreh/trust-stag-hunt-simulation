"""Partner strategies for agent interactions."""

from .base import BasePartner

# Fixed strategies
from .fixed import (
    AlwaysCooperatePartner,
    AlwaysDefectPartner,
    RandomPartner,
    PeriodicCheaterPartner,
    SingleCyclePartner,
    GradualDeteriorationPartner,
    BreakRepairPartner,
    ReliabilityRampPartner,
)

# Reactive strategies  
from .reactive import (
    TitForTatCooperatePartner,
    GrimTriggerPartner,
    SuspiciousTitForTatPartner,
    PavlovPartner,
    TitForTatDefectPartner,
)

# Belief-driven strategies (mirror focal agent dynamics)
from .belief_driven import (
    BeliefDrivenPartnerBase,
    StrategicCheaterPartner,
    DistrustLockPartner,
    ExpectationViolationPartner,
    AdaptivePartner,
)

from .learners import (
    CoAdaptivePartner,
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

from .registry import (
    MANUSCRIPT_PARTNERS,
    RETIRED_PARTNERS,
    ALL_PARTNERS,
    DIAGNOSTIC_PARTNERS,
    COLLABORATIVE_PARTNERS,
    HELDOUT_PARTNERS,
    PARTNER_CATEGORY,
    build_partner,
    partner_factory,
    partner_factories,
)

__all__ = [
    'BasePartner',
    # Fixed
    'AlwaysCooperatePartner',
    'AlwaysDefectPartner', 
    'RandomPartner',
    'PeriodicCheaterPartner',
    'SingleCyclePartner',
    'GradualDeteriorationPartner',
    'BreakRepairPartner',
    'ReliabilityRampPartner',
    # Reactive
    'TitForTatCooperatePartner',
    'GrimTriggerPartner',
    'SuspiciousTitForTatPartner',
    'PavlovPartner',
    'TitForTatDefectPartner',
    # Belief-driven
    'BeliefDrivenPartnerBase',
    'StrategicCheaterPartner',
    'DistrustLockPartner',
    'ExpectationViolationPartner',
    'AdaptivePartner',
    'CoAdaptivePartner',
    'MyopicBestResponsePartner',
    'EWAPartner',
    'AspirationPartner',
    'LenientPartner',
    'AdaptivePlayPartner',
    'COLLABORATIVE_PARTNERS',
    'GenerousTitForTatPartner',
    'ReformerPartner',
    'BreachPartner',
    'RegimeSwitchPartner',
    'HELDOUT_PARTNERS',
    'MANUSCRIPT_PARTNERS',
    'RETIRED_PARTNERS',
    'ALL_PARTNERS',
    'DIAGNOSTIC_PARTNERS',
    'PARTNER_CATEGORY',
    'build_partner',
    'partner_factory',
    'partner_factories',
]
