"""Underflow-safe Q16.16 negative-log desirability routing.

Model quantization is host-side. Sweeps and deterministic route extraction use
integers only; a small softplus LUT replaces runtime logarithms/exponentials.
INT32_MAX denotes unreachable states, not a large but finite cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from mcu import INT32_MAX, Q16_SCALE, MCUArena

if TYPE_CHECKING:
    from ember.routing import LMDPModel

LUT_STEP = 256  # 1/256 in Q16.16
LUT_LIMIT = 16 * Q16_SCALE
# This table is also exported verbatim to firmware/log_lmdp_lut.h.
SOFTPLUS_LUT = np.rint(
    np.logaddexp(0.0, -np.arange(LUT_LIMIT // LUT_STEP + 1) / 256.0) * Q16_SCALE
).astype(np.int32)
SOFTPLUS_LUT.setflags(write=False)


def softplus_q16(distance: int) -> int:
    """Rounded linear interpolation of log(1 + exp(-distance))."""
    if distance < 0:
        raise ValueError("softplus distance must be nonnegative")
    if distance >= LUT_LIMIT:
        return 0
    index, remainder = divmod(distance, LUT_STEP)
    return (
        int(SOFTPLUS_LUT[index]) * (LUT_STEP - remainder)
        + int(SOFTPLUS_LUT[index + 1]) * remainder
        + LUT_STEP // 2
    ) // LUT_STEP


def _array(arena: MCUArena, values: np.ndarray, dtype) -> np.ndarray:
    source = np.asarray(values, dtype=dtype)
    output = np.frombuffer(arena.alloc(source.nbytes), dtype=dtype).reshape(
        source.shape
    )
    output[:] = source
    return output


@dataclass(frozen=True, slots=True)
class LogQ16Model:
    row_ptr: np.ndarray
    col_idx: np.ndarray
    negative_log_passive: np.ndarray
    cost_over_temperature: np.ndarray
    exit_mask: np.ndarray
    death_mask: np.ndarray

    @classmethod
    def from_float(cls, model: LMDPModel, arena: MCUArena) -> LogQ16Model:
        log_passive = np.full(model.passive_prob.shape, INT32_MAX, dtype=np.int64)
        positive = model.passive_prob > 0.0
        encoded = np.rint(-np.log(model.passive_prob[positive]) * Q16_SCALE)
        costs = np.rint(model.state_cost / model.config.control_temperature * Q16_SCALE)
        if np.any(encoded >= INT32_MAX) or np.any(costs >= INT32_MAX):
            raise ValueError("model exceeds log-Q16 finite range")
        nonterminal = ~(model.exit_mask | model.death_mask)
        if np.any(costs[nonterminal] < 1):
            raise ValueError("positive state costs must survive Q16 quantization")
        log_passive[positive] = encoded.astype(np.int64)
        return cls(
            _array(arena, model.row_ptr, np.int32),
            _array(arena, model.col_idx, np.int32),
            _array(arena, log_passive, np.int32),
            _array(arena, costs, np.int32),
            _array(arena, model.exit_mask, np.uint8),
            _array(arena, model.death_mask, np.uint8),
        )

    @property
    def n_states(self) -> int:
        return int(self.cost_over_temperature.size)


def log_q16_sweep(model: LogQ16Model, current: np.ndarray, output: np.ndarray) -> None:
    """Write one synchronous integer soft-min sweep into a separate buffer."""
    if current.shape != (model.n_states,) or output.shape != current.shape:
        raise ValueError("state buffers must have length N")
    if current.dtype != np.int32 or output.dtype != np.int32:
        raise ValueError("state buffers must be int32")
    if np.shares_memory(current, output):
        raise ValueError("Jacobi state buffers must not alias")
    for state in range(model.n_states):
        if model.exit_mask[state]:
            output[state] = 0
            continue
        if model.death_mask[state]:
            output[state] = INT32_MAX
            continue
        aggregate = INT32_MAX
        for edge in range(int(model.row_ptr[state]), int(model.row_ptr[state + 1])):
            value = int(current[int(model.col_idx[edge])])
            passive = int(model.negative_log_passive[edge])
            if value == INT32_MAX or passive == INT32_MAX:
                continue
            term = min(INT32_MAX, value + passive)
            if aggregate == INT32_MAX:
                aggregate = term
            elif term != INT32_MAX:
                aggregate = min(aggregate, term) - softplus_q16(abs(aggregate - term))
        output[state] = (
            INT32_MAX
            if aggregate == INT32_MAX
            else min(
                INT32_MAX, max(0, aggregate + int(model.cost_over_temperature[state]))
            )
        )


@dataclass(frozen=True, slots=True)
class LogQ16Solution:
    model: LogQ16Model
    negative_log_desirability: np.ndarray
    sweeps: int
    fixed_point: bool
    arena_high_water_bytes: int

    def route(self, start: int) -> tuple[int, ...]:
        """Select the most likely strictly progressing action using integer scores."""
        model = self.model
        if not 0 <= start < model.n_states:
            raise ValueError("start is outside the state space")
        values = self.negative_log_desirability
        if values[start] == INT32_MAX or model.death_mask[start]:
            return ()
        route = [start]
        for _ in range(model.n_states):
            state = route[-1]
            if model.exit_mask[state]:
                return tuple(route)
            choices = []
            for edge in range(int(model.row_ptr[state]), int(model.row_ptr[state + 1])):
                target = int(model.col_idx[edge])
                passive = int(model.negative_log_passive[edge])
                if passive != INT32_MAX and values[target] < values[state]:
                    choices.append((int(values[target]) + passive, target))
            if not choices:
                return ()
            route.append(min(choices)[1])
        return ()

    def desirability_reference(self) -> np.ndarray:
        """Decode for host diagnostics; deployed policy never exponentiates values."""
        result = np.exp(-self.negative_log_desirability.astype(np.float64) / Q16_SCALE)
        result[self.negative_log_desirability == INT32_MAX] = 0.0
        return result


def solve_log_q16(
    model: LMDPModel,
    *,
    arena: MCUArena | None = None,
    _sweep=None,
) -> LogQ16Solution:
    """Iterate to an integer fixed point with arena-backed CSR and state buffers."""
    storage = arena if arena is not None else MCUArena(200 * 1024)
    quantized = LogQ16Model.from_float(model, storage)
    current = _array(storage, np.full(model.n_states, INT32_MAX), np.int32)
    following = _array(storage, np.empty(model.n_states, dtype=np.int32), np.int32)
    current[model.exit_mask] = 0
    fixed = False
    sweeps = 0
    for sweep in range(1, model.config.max_sweeps + 1):
        sweeps = sweep
        (_sweep or log_q16_sweep)(quantized, current, following)
        fixed = bool(np.array_equal(current, following))
        current, following = following, current
        if fixed:
            break
    return LogQ16Solution(quantized, current, sweeps, fixed, storage.high_water)
