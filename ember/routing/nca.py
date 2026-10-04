"""Safeguarded int8 neural cellular accelerator for the LMDP fixed point.

The learned/local proposal is never trusted directly. For a contraction T and
current iterate z, the classical candidate is j = T(z). The proposal p is accepted
only when its Bellman residual is no larger than the residual of j. Since the
classical residual contracts by alpha < 1, arbitrary weights cannot make the
returned iteration diverge.

This module is the NumPy reference kernel. It uses int8 parameter tables and integer
matrix multiplies for the NCA proposal; the Bellman operator and safeguard remain
float64 so their result can be compared with the Phase-1 oracle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .lmdp import (
    LMDPModel,
    LMDPSolution,
    _passive_matvec,
    _scaled_residual,
    _states_reaching_exit,
)

_Q7 = 127


def _readonly(values: np.ndarray, dtype: np.dtype[np.generic]) -> np.ndarray:
    out = np.array(values, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class Int8NCAWeights:
    """A 2,465-byte graph-NCA parameter set."""

    input_weight: np.ndarray
    input_bias: np.ndarray
    hidden_weight: np.ndarray
    hidden_bias: np.ndarray
    output_weight: np.ndarray
    output_bias: int = 0

    def __post_init__(self) -> None:
        iw = _readonly(self.input_weight, np.dtype(np.int8))
        ib = _readonly(self.input_bias, np.dtype(np.int8))
        hw = _readonly(self.hidden_weight, np.dtype(np.int8))
        hb = _readonly(self.hidden_bias, np.dtype(np.int8))
        ow = _readonly(self.output_weight, np.dtype(np.int8))
        if iw.shape != (32, 8):
            raise ValueError("input_weight must have shape (32, 8)")
        if ib.shape != (32,):
            raise ValueError("input_bias must have shape (32,)")
        if hw.shape != (64, 32):
            raise ValueError("hidden_weight must have shape (64, 32)")
        if hb.shape != (64,):
            raise ValueError("hidden_bias must have shape (64,)")
        if ow.shape != (64,):
            raise ValueError("output_weight must have shape (64,)")
        object.__setattr__(self, "input_weight", iw)
        object.__setattr__(self, "input_bias", ib)
        object.__setattr__(self, "hidden_weight", hw)
        object.__setattr__(self, "hidden_bias", hb)
        object.__setattr__(self, "output_weight", ow)
        object.__setattr__(self, "output_bias", np.int8(self.output_bias))

    @property
    def parameter_bytes(self) -> int:
        return int(
            self.input_weight.nbytes
            + self.input_bias.nbytes
            + self.hidden_weight.nbytes
            + self.hidden_bias.nbytes
            + self.output_weight.nbytes
            + np.asarray(self.output_bias).nbytes
        )

    @classmethod
    def default(cls) -> Int8NCAWeights:
        iw = np.zeros((32, 8), dtype=np.int8)
        ib = np.zeros(32, dtype=np.int8)
        hw = np.zeros((64, 32), dtype=np.int8)
        hb = np.zeros(64, dtype=np.int8)
        ow = np.zeros(64, dtype=np.int8)
        iw[0, 2] = 127
        iw[1, 2] = -127
        hw[0, 0] = 64
        hw[1, 1] = 64
        ow[0] = 127
        ow[1] = -127
        return cls(iw, ib, hw, hb, ow)


@dataclass(frozen=True, slots=True)
class SafeguardedNCAResult:
    solution: LMDPSolution
    accepted_proposals: int
    rejected_proposals: int
    parameter_bytes: int
    worst_residual_ratio: float

    def __post_init__(self) -> None:
        if self.accepted_proposals < 0 or self.rejected_proposals < 0:
            raise ValueError("proposal counts must be nonnegative")
        if self.parameter_bytes <= 0:
            raise ValueError("parameter_bytes must be positive")
        if self.worst_residual_ratio < 0.0 or not math.isfinite(
            self.worst_residual_ratio
        ):
            raise ValueError("worst_residual_ratio must be finite and nonnegative")


def _bellman(model: LMDPModel, values: np.ndarray) -> np.ndarray:
    attenuation = np.exp(-model.state_cost / model.config.control_temperature)
    candidate = attenuation * _passive_matvec(model, values)
    candidate[model.exit_mask] = 1.0
    candidate[model.death_mask] = 0.0
    return candidate


def _neighbour_features(
    model: LMDPModel,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = _passive_matvec(model, values)
    maximum = np.empty(model.n_states, dtype=np.float64)
    for state in range(model.n_states):
        row = model.row(state)
        maximum[state] = float(np.max(values[model.col_idx[row]]))
    return mean, maximum


def nca_proposal(
    model: LMDPModel,
    current: np.ndarray,
    classical: np.ndarray,
    weights: Int8NCAWeights,
) -> np.ndarray:
    """Produce one local graph-NCA proposal using Q0.7 activations."""

    z = np.asarray(current, dtype=np.float64)
    j = np.asarray(classical, dtype=np.float64)
    if z.shape != (model.n_states,) or j.shape != z.shape:
        raise ValueError("NCA fields must have one value per state")
    neighbour_mean, neighbour_max = _neighbour_features(model, z)
    residual = j - z
    attenuation = np.exp(-model.state_cost / model.config.control_temperature)
    features = np.column_stack(
        (
            z,
            j,
            residual,
            np.abs(residual),
            neighbour_mean,
            neighbour_max,
            attenuation,
            np.ones(model.n_states),
        )
    )
    input_q7 = np.clip(np.rint(features * _Q7), -_Q7, _Q7).astype(np.int16)
    hidden1 = (
        input_q7.astype(np.int32) @ weights.input_weight.astype(np.int32).T
    ) // _Q7
    hidden1 += weights.input_bias.astype(np.int32)
    hidden1 = np.clip(hidden1, 0, _Q7)
    hidden2 = (
        hidden1 @ weights.hidden_weight.astype(np.int32).T
    ) // _Q7
    hidden2 += weights.hidden_bias.astype(np.int32)
    hidden2 = np.clip(hidden2, 0, _Q7)
    delta_q7 = (
        hidden2 @ weights.output_weight.astype(np.int32)
    ) // _Q7 + int(weights.output_bias)
    proposal = np.clip(j + delta_q7.astype(np.float64) / _Q7, 0.0, 1.0)
    proposal[model.exit_mask] = 1.0
    proposal[model.death_mask] = 0.0
    return proposal


def _solution(model: LMDPModel, desirability: np.ndarray, sweeps: int) -> LMDPSolution:
    fixed_point = _bellman(model, desirability)
    reachable = _states_reaching_exit(model)
    relevant = reachable & ~(model.exit_mask | model.death_mask)
    residual = float(np.max(np.abs(fixed_point - desirability)))
    relative = _scaled_residual(fixed_point, desirability, relevant)
    converged = (
        np.all(desirability[reachable] > 0.0)
        and residual <= model.config.tolerance
        and relative <= model.config.tolerance
    )
    value = np.full(model.n_states, np.inf, dtype=np.float64)
    positive = desirability > 0.0
    value[positive] = (
        -model.config.control_temperature * np.log(desirability[positive])
    )
    value[model.exit_mask] = 0.0
    numerator = model.passive_prob * desirability[model.col_idx]
    denominator = np.add.reduceat(numerator, model.row_ptr[:-1])
    policy = np.zeros_like(numerator)
    for state in range(model.n_states):
        row = model.row(state)
        if model.death_mask[state] or denominator[state] <= 0.0:
            continue
        policy[row] = numerator[row] / denominator[state]
    return LMDPSolution(
        desirability=desirability,
        value=value,
        policy_prob=policy,
        sweeps=sweeps,
        residual=residual,
        relative_residual=relative,
        converged=converged,
    )


def solve_safeguarded_nca(
    model: LMDPModel,
    weights: Int8NCAWeights | None = None,
) -> SafeguardedNCAResult:
    """Solve with arbitrary int8 proposals and a contraction safeguard."""

    table = weights or Int8NCAWeights.default()
    current = np.zeros(model.n_states, dtype=np.float64)
    current[model.exit_mask] = 1.0
    reachable = _states_reaching_exit(model)
    relevant = reachable & ~(model.exit_mask | model.death_mask)
    accepted = rejected = 0
    worst_ratio = 0.0
    sweeps = 0

    for sweep in range(1, model.config.max_sweeps + 1):
        sweeps = sweep
        classical = _bellman(model, current)
        proposal = nca_proposal(model, current, classical, table)
        current_residual = float(np.max(np.abs(classical - current)))
        classical_residual = float(
            np.max(np.abs(_bellman(model, classical) - classical))
        )
        proposal_residual = float(
            np.max(np.abs(_bellman(model, proposal) - proposal))
        )
        safe_bound = model.contraction_bound * current_residual
        if (
            np.all(np.isfinite(proposal))
            and proposal_residual <= classical_residual + 1.0e-15
            and proposal_residual <= safe_bound + 1.0e-15
        ):
            following = proposal
            accepted += 1
            next_residual = proposal_residual
        else:
            following = classical
            rejected += 1
            next_residual = classical_residual
        if current_residual > 0.0:
            worst_ratio = max(worst_ratio, next_residual / current_residual)
        next_fixed = _bellman(model, following)
        absolute = float(np.max(np.abs(next_fixed - following)))
        relative = _scaled_residual(next_fixed, following, relevant)
        current = following
        if (
            np.all(current[reachable] > 0.0)
            and absolute <= model.config.tolerance
            and relative <= model.config.tolerance
        ):
            break

    return SafeguardedNCAResult(
        solution=_solution(model, current, sweeps),
        accepted_proposals=accepted,
        rejected_proposals=rejected,
        parameter_bytes=table.parameter_bytes,
        worst_residual_ratio=worst_ratio,
    )


__all__ = [
    "Int8NCAWeights",
    "SafeguardedNCAResult",
    "nca_proposal",
    "solve_safeguarded_nca",
]
