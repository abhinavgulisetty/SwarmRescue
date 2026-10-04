"""Classical and learned fixed-point routing kernels."""

from .lmdp import (
    LMDPConfig,
    LMDPModel,
    LMDPSolution,
    OptimismDiagnostic,
    OracleCostConfig,
    build_lmdp,
    greedy_route,
    oracle_state_cost,
    solve_jacobi,
    trajectory_optimism,
)
from .nca import (
    Int8NCAWeights,
    SafeguardedNCAResult,
    nca_proposal,
    solve_safeguarded_nca,
)

__all__ = [
    "Int8NCAWeights",
    "LMDPConfig",
    "LMDPModel",
    "LMDPSolution",
    "OptimismDiagnostic",
    "OracleCostConfig",
    "SafeguardedNCAResult",
    "build_lmdp",
    "greedy_route",
    "nca_proposal",
    "oracle_state_cost",
    "solve_jacobi",
    "solve_safeguarded_nca",
    "trajectory_optimism",
]
