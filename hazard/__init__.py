"""Ground-truth fire and smoke dynamics for EMBER Phase 1.

The hazard state belongs to the simulator, never to a robot.  Robot-side
beliefs consume sparse sensor samples derived from this state in later phases.
"""

from .solver import (
    HazardConfig,
    HazardDomain,
    HazardSolver,
    HazardState,
    transport_periodic_1d,
)

__all__ = [
    "HazardConfig",
    "HazardDomain",
    "HazardSolver",
    "HazardState",
    "transport_periodic_1d",
]
