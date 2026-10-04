"""Adjoint-weighted value-of-information scheduling for Gaussian packets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve

from ember.comms import (
    MAX_UPDATES,
    PACKET_HEADER_BYTES,
    UPDATE_BYTES,
    GaussianPacket,
    packetize_gaussians,
)
from ember.routing.lmdp import LMDPModel


@dataclass(frozen=True, slots=True)
class AdjointSolution:
    field: np.ndarray
    sweeps: int
    residual: float
    converged: bool

    def __post_init__(self) -> None:
        values = np.array(self.field, dtype=np.float64, copy=True)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("adjoint field must be a finite vector")
        if np.any(values < 0.0) or self.sweeps < 0:
            raise ValueError("adjoint values and sweep count must be nonnegative")
        if self.residual < 0.0 or not math.isfinite(self.residual):
            raise ValueError("adjoint residual must be finite and nonnegative")
        values.setflags(write=False)
        object.__setattr__(self, "field", values)


def _transpose_step(
    model: LMDPModel,
    values: np.ndarray,
    source_weight: np.ndarray,
) -> np.ndarray:
    attenuation = np.exp(-model.state_cost / model.config.control_temperature)
    following = np.array(source_weight, dtype=np.float64, copy=True)
    for source in range(model.n_states):
        if model.exit_mask[source] or model.death_mask[source]:
            continue
        row = model.row(source)
        contribution = attenuation[source] * values[source] * model.passive_prob[row]
        np.add.at(following, model.col_idx[row], contribution)
    return following


def solve_adjoint(
    model: LMDPModel,
    source_weight: np.ndarray,
    *,
    tolerance: float | None = None,
    max_sweeps: int | None = None,
) -> AdjointSolution:
    """Solve the transposed desirability system for route relevance."""

    source = np.asarray(source_weight, dtype=np.float64)
    if source.shape != (model.n_states,):
        raise ValueError("source_weight must have one value per state")
    if not np.all(np.isfinite(source)) or np.any(source < 0.0):
        raise ValueError("source_weight must be finite and nonnegative")
    if not np.any(source > 0.0):
        raise ValueError("source_weight must contain positive route relevance")
    tol = model.config.tolerance if tolerance is None else tolerance
    cap = model.config.max_sweeps if max_sweeps is None else max_sweeps
    if not math.isfinite(tol) or tol <= 0.0 or cap < 1:
        raise ValueError("adjoint tolerance and sweep cap must be positive")

    values = np.zeros(model.n_states, dtype=np.float64)
    residual = math.inf
    converged = False
    sweeps = 0
    for sweep in range(1, cap + 1):
        sweeps = sweep
        following = _transpose_step(model, values, source)
        residual = float(np.max(np.abs(following - values)))
        values = following
        if residual <= tol:
            converged = True
            break
    fixed = _transpose_step(model, values, source)
    residual = float(np.max(np.abs(fixed - values)))
    converged = residual <= tol
    return AdjointSolution(values, sweeps, residual, converged)


def solve_adjoint_sparse(model, source_weight):
    """Host sparse solve of the same transposed operator, with a residual check.

    The zero sweep count denotes a direct solve, not a Jacobi speedup or MCU
    timing result. Terminal rows do not emit flow, matching `_transpose_step`.
    """
    source = np.asarray(source_weight, dtype=float)
    if source.shape != (model.n_states,) or not np.all(np.isfinite(source)):
        raise ValueError("source must be finite with one value per state")
    if np.any(source < 0) or not np.any(source > 0):
        raise ValueError("source must be nonnegative and nonzero")
    attenuation = np.exp(-model.state_cost / model.config.control_temperature)
    attenuation[model.exit_mask | model.death_mask] = 0
    row_states = np.repeat(np.arange(model.n_states), np.diff(model.row_ptr))
    transition = csr_matrix(
        (attenuation[row_states] * model.passive_prob, model.col_idx, model.row_ptr),
        shape=(model.n_states, model.n_states),
    )
    operator = eye(model.n_states, format="csr") - transition.T
    values = np.asarray(spsolve(operator, source))
    if not np.all(np.isfinite(values)) or np.any(values < -1e-10):
        raise RuntimeError("physical adjoint sparse solve failed")
    values = np.maximum(values, 0)
    residual = float(np.max(np.abs(operator @ values - source)))
    return AdjointSolution(values, 0, residual, residual <= model.config.tolerance)


@dataclass(slots=True)
class TokenBucket:
    """Integer-byte token bucket suitable for a per-tick MCU scheduler."""

    capacity_bytes: int
    refill_bytes: int
    tokens: int = field(init=False)

    def __post_init__(self) -> None:
        if self.capacity_bytes <= 0 or self.refill_bytes < 0:
            raise ValueError("token bucket sizes must be positive/nonnegative")
        self.tokens = self.capacity_bytes

    def refill(self) -> None:
        self.tokens = min(self.capacity_bytes, self.tokens + self.refill_bytes)

    def consume(self, amount: int) -> bool:
        if amount < 0:
            raise ValueError("token consumption must be nonnegative")
        if amount > self.tokens:
            return False
        self.tokens -= amount
        return True


@dataclass(frozen=True, slots=True)
class VoISchedule:
    packets: tuple[GaussianPacket, ...]
    selected_cells: tuple[int, ...]
    scores: tuple[float, ...]
    information_gain_nats: float
    bytes_scheduled: int

    @property
    def nats_per_byte(self) -> float:
        if self.bytes_scheduled == 0:
            return 0.0
        return self.information_gain_nats / self.bytes_scheduled


def gaussian_kl(
    prior_mean: np.ndarray,
    prior_variance: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_variance: np.ndarray,
) -> np.ndarray:
    """Return KL(posterior || prior) for independent scalar Gaussians."""

    pm = np.asarray(prior_mean, dtype=np.float64)
    pv = np.asarray(prior_variance, dtype=np.float64)
    qm = np.asarray(posterior_mean, dtype=np.float64)
    qv = np.asarray(posterior_variance, dtype=np.float64)
    if not (pm.shape == pv.shape == qm.shape == qv.shape):
        raise ValueError("Gaussian vectors must have matching shapes")
    if (
        not np.all(np.isfinite(pm))
        or not np.all(np.isfinite(qm))
        or not np.all(np.isfinite(pv))
        or not np.all(np.isfinite(qv))
        or np.any(pv <= 0.0)
        or np.any(qv <= 0.0)
    ):
        raise ValueError("Gaussian parameters must be finite with positive variance")
    return 0.5 * (np.log(pv / qv) + (qv + np.square(qm - pm)) / pv - 1.0)


def schedule_gaussian_updates(
    *,
    sender: int,
    sequence: int,
    cells: np.ndarray,
    prior_mean: np.ndarray,
    prior_variance: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_variance: np.ndarray,
    adjoint: np.ndarray,
    bucket: TokenBucket,
    ttl: int = 4,
    minimum_score: float = 0.0,
) -> VoISchedule:
    """Greedily schedule the highest adjoint-weighted KL updates."""

    cell_ids = np.asarray(cells)
    if cell_ids.ndim != 1 or not np.issubdtype(cell_ids.dtype, np.integer):
        raise ValueError("cells must be a one-dimensional integer vector")
    arrays = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            prior_mean,
            prior_variance,
            posterior_mean,
            posterior_variance,
        )
    )
    if any(values.shape != cell_ids.shape for values in arrays):
        raise ValueError("Gaussian update arrays must match cells")
    relevance = np.asarray(adjoint, dtype=np.float64)
    if (
        relevance.ndim != 1
        or np.any(cell_ids < 0)
        or np.any(cell_ids >= relevance.size)
    ):
        raise ValueError("cells lie outside the adjoint field")
    if not np.all(np.isfinite(relevance)) or np.any(relevance < 0.0):
        raise ValueError("adjoint relevance must be finite and nonnegative")
    if not math.isfinite(minimum_score) or minimum_score < 0.0:
        raise ValueError("minimum_score must be finite and nonnegative")

    kl = gaussian_kl(*arrays)
    score = kl * relevance[cell_ids]
    order = np.lexsort((cell_ids, -score))
    selected: list[int] = []
    committed = 0
    for index in order:
        if score[index] <= minimum_score:
            continue
        extra = UPDATE_BYTES
        if len(selected) % MAX_UPDATES == 0:
            extra += PACKET_HEADER_BYTES
        if committed + extra > bucket.tokens:
            continue
        selected.append(int(index))
        committed += extra
    if not selected:
        return VoISchedule((), (), (), 0.0, 0)

    chosen = np.asarray(selected, dtype=np.int64)
    packets = packetize_gaussians(
        sender,
        sequence,
        cell_ids[chosen],
        arrays[2][chosen],
        arrays[3][chosen],
        ttl=ttl,
    )
    actual_bytes = sum(packet.size_bytes for packet in packets)
    if actual_bytes != committed or not bucket.consume(actual_bytes):
        raise RuntimeError("token accounting disagrees with packet encoding")
    selected_cells = tuple(int(value) for value in cell_ids[chosen])
    selected_scores = tuple(float(value) for value in score[chosen])
    return VoISchedule(
        packets=packets,
        selected_cells=selected_cells,
        scores=selected_scores,
        information_gain_nats=float(np.sum(kl[chosen])),
        bytes_scheduled=actual_bytes,
    )


__all__ = [
    "AdjointSolution",
    "TokenBucket",
    "VoISchedule",
    "gaussian_kl",
    "schedule_gaussian_updates",
    "solve_adjoint",
]


def solve_policy_occupancy(model, negative_log_desirability, source_weight):
    """Expected route visitation under the exit-conditioned local policy.

    Unlike the killed passive adjoint, this field retains downstream relevance
    when the source-to-exit desirability is tiny. Terminal states absorb flow.
    Host sparse solve is a communication reference, not an embedded timing claim.
    """
    from mcu import INT32_MAX, Q16_SCALE

    values = np.asarray(negative_log_desirability)
    source = np.asarray(source_weight, dtype=float)
    if values.shape != (model.n_states,) or source.shape != values.shape:
        raise ValueError("occupancy inputs must have one value per state")
    if np.any(source < 0) or not np.all(np.isfinite(source)) or not np.any(source > 0):
        raise ValueError("occupancy source must be finite and positive")
    probabilities = np.zeros_like(model.passive_prob)
    for state in range(model.n_states):
        if (
            model.exit_mask[state]
            or model.death_mask[state]
            or values[state] == INT32_MAX
        ):
            continue
        row = model.row(state)
        targets = model.col_idx[row]
        positive = (model.passive_prob[row] > 0) & (values[targets] != INT32_MAX)
        if not np.any(positive):
            continue
        logits = np.full(targets.size, -np.inf)
        logits[positive] = (
            np.log(model.passive_prob[row][positive])
            - values[targets[positive]].astype(float) / Q16_SCALE
        )
        logits -= np.max(logits)
        weights = np.exp(logits)
        probabilities[row] = weights / np.sum(weights)
    transition = csr_matrix(
        (probabilities, model.col_idx, model.row_ptr),
        shape=(model.n_states, model.n_states),
    )
    operator = eye(model.n_states, format="csr") - transition.T
    field = np.asarray(spsolve(operator, source))
    if not np.all(np.isfinite(field)) or np.any(field < -1e-8):
        raise RuntimeError("policy occupancy solve failed")
    residual = float(np.max(np.abs(operator @ field - source)))
    if residual > 1e-8 * max(1.0, float(np.max(source))):
        raise RuntimeError("policy occupancy residual exceeded tolerance")
    return np.maximum(field, 0.0)


def schedule_coverage_updates(
    *,
    sender,
    sequence,
    cells,
    prior_mean,
    prior_variance,
    posterior_mean,
    posterior_variance,
    relevance,
    positions_m,
    bucket,
    length_scale_m=2.0,
):
    """Greedy submodular spatial coverage of route-weighted information.

    Nearby redundant observations compete for coverage rather than occupying
    every scarce packet slot. A relevance floor keeps hazard evidence outside
    the estimated route eligible. The source uses only its own sensor factors.
    """
    cells = np.asarray(cells, dtype=np.int64)
    positions = np.asarray(positions_m, dtype=float)
    relevance = np.asarray(relevance, dtype=float)
    if length_scale_m <= 0 or not math.isfinite(length_scale_m):
        raise ValueError("coverage length scale must be finite and positive")
    if positions.shape != (relevance.size, 3):
        raise ValueError("coverage positions and relevance must describe one graph")
    kl = gaussian_kl(prior_mean, prior_variance, posterior_mean, posterior_variance)
    weights = kl * (relevance[cells] + 0.05 * max(1.0, float(np.max(relevance))))
    covered = np.zeros(cells.size)
    selected = []
    committed = 0
    while len(selected) < cells.size:
        extra = UPDATE_BYTES + (
            PACKET_HEADER_BYTES if len(selected) % MAX_UPDATES == 0 else 0
        )
        if committed + extra > bucket.tokens:
            break
        best, best_gain, best_cover = -1, 0.0, None
        for index in range(cells.size):
            if index in selected:
                continue
            delta = positions[cells] - positions[cells[index]]
            cover = np.exp(-np.sum(delta * delta, axis=1) / (2 * length_scale_m**2))
            gain = float(np.dot(weights, np.maximum(cover - covered, 0.0)))
            if gain > best_gain or (
                gain == best_gain and best >= 0 and cells[index] < cells[best]
            ):
                best, best_gain, best_cover = index, gain, cover
        if best < 0:
            break
        selected.append(best)
        covered = np.maximum(covered, best_cover)
        committed += extra
    chosen = np.array(selected, dtype=np.int64)
    return schedule_gaussian_updates(
        sender=sender,
        sequence=sequence,
        cells=cells[chosen],
        prior_mean=np.asarray(prior_mean)[chosen],
        prior_variance=np.asarray(prior_variance)[chosen],
        posterior_mean=np.asarray(posterior_mean)[chosen],
        posterior_variance=np.asarray(posterior_variance)[chosen],
        adjoint=np.ones(relevance.size),
        bucket=bucket,
    )
