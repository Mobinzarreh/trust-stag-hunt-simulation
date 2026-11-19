"""Simulation engine for agent-partner interactions."""

from .runner import run_agent_simulation, run_paired_simulation, slot_rng, AGENT_SLOT_SEED_OFFSET

__all__ = ['run_agent_simulation', 'run_paired_simulation', 'slot_rng', 'AGENT_SLOT_SEED_OFFSET']
