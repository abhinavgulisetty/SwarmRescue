"""Asynchronous scalar Gaussian belief propagation on a traversability graph.

Each cell owns a Gaussian local factor and every graph edge owns a smoothness
factor ``x_j = x_i + N(0, edge_variance)``.  Messages use canonical
information/precision form, matching EMBER's radio packet representation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from cascade.extract import TraversabilityGraph
from ember.belief import GaussianBelief


@dataclass(frozen=True, slots=True)
class GBPConfig:
    edge_variance: float = 0.5
    damping: float = 0.35
    tolerance: float = 1.0e-6
    max_sweeps: int = 500

    def __post_init__(self) -> None:
        if self.edge_variance <= 0.0 or not math.isfinite(self.edge_variance):
            raise ValueError("edge_variance must be finite and positive")
        if not 0.0 < self.damping <= 1.0 or not math.isfinite(self.damping):
            raise ValueError("damping must be finite and in (0, 1]")
        if self.tolerance <= 0.0 or not math.isfinite(self.tolerance):
            raise ValueError("tolerance must be finite and positive")
        if self.max_sweeps < 1:
            raise ValueError("max_sweeps must be positive")


@dataclass(frozen=True, slots=True)
class GBPResult:
    belief: GaussianBelief
    sweeps: int
    residual: float
    converged: bool


def topology_index_dtype(count: int) -> np.dtype:
    """Smallest supported unsigned index for ``count`` zero-based entries.

    Endpoints and reverse-message indices have independent bounds. Check the
    bound before casting so the 65,536-entry boundary cannot silently wrap.
    """
    if not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("index count must be a nonnegative integer")
    for dtype in (np.uint16, np.uint32, np.uint64):
        if count <= int(np.iinfo(dtype).max) + 1:
            return np.dtype(dtype)
    raise ValueError("index count exceeds uint64 range")


def gbp_storage_bytes(nodes: int, edges: int) -> int:
    """Aligned retained numeric GBP bytes, excluding host graph and scratch."""
    endpoint_bytes = topology_index_dtype(nodes).itemsize
    reverse_bytes = topology_index_dtype(2 * edges).itemsize
    sizes = (
        8 * nodes, 8 * nodes,
        2 * edges * endpoint_bytes, 2 * edges * endpoint_bytes,
        2 * edges * reverse_bytes,
        16 * edges, 16 * edges, 16 * edges,
    )
    used = 0
    for size in sizes:
        used = (used + 7) & ~7
        used += size
    return used


class GaussianBeliefPropagation:
    """Loopy GBP with deterministic or caller-selected asynchronous updates."""

    def __init__(
        self,
        graph: TraversabilityGraph,
        *,
        mean: float,
        variance: float,
        config: GBPConfig | None = None,
    ) -> None:
        if not math.isfinite(mean):
            raise ValueError("prior mean must be finite")
        if variance <= 0.0 or not math.isfinite(variance):
            raise ValueError("prior variance must be finite and positive")
        self.graph = graph
        self.config = config or GBPConfig()
        self.local_precision = np.full(graph.n_nodes, 1.0 / variance)
        self.local_information = np.full(graph.n_nodes, mean / variance)

        edges = np.asarray(graph.edges)
        if edges.size and (
            edges.ndim != 2 or edges.shape[1] != 2
            or not np.issubdtype(edges.dtype, np.integer)
            or np.any(edges < 0) or np.any(edges >= graph.n_nodes)
        ):
            raise ValueError("graph edges must have valid integer endpoints")
        endpoint_dtype = topology_index_dtype(graph.n_nodes)
        reverse_dtype = topology_index_dtype(2 * graph.n_edges)
        if edges.size:
            self.source = np.concatenate((edges[:, 0], edges[:, 1])).astype(
                endpoint_dtype
            )
            self.target = np.concatenate((edges[:, 1], edges[:, 0])).astype(
                endpoint_dtype
            )
            count = edges.shape[0]
            self.reverse = np.concatenate(
                (np.arange(count, 2 * count), np.arange(count))
            ).astype(reverse_dtype)
            difficulty = np.asarray(graph.edge_cost, dtype=np.float64)
            scale = difficulty / max(float(np.median(difficulty)), 1.0e-9)
            scale = np.maximum(scale, 1.0e-6)
            self.edge_variance = self.config.edge_variance * np.concatenate(
                (scale, scale)
            )
        else:
            self.source = np.zeros(0, dtype=endpoint_dtype)
            self.target = np.zeros(0, dtype=endpoint_dtype)
            self.reverse = np.zeros(0, dtype=reverse_dtype)
            self.edge_variance = np.zeros(0)
        self.message_precision = np.zeros(self.source.size)
        self.message_information = np.zeros(self.source.size)

    def observe(
        self,
        cells: np.ndarray,
        mean: np.ndarray,
        variance: np.ndarray,
    ) -> None:
        indices = np.asarray(cells)
        means = np.asarray(mean, dtype=np.float64)
        variances = np.asarray(variance, dtype=np.float64)
        if (
            indices.ndim != 1
            or indices.shape != means.shape
            or means.shape != variances.shape
        ):
            raise ValueError("observation vectors must have matching shapes")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("observation cells must be integers")
        if np.any((indices < 0) | (indices >= self.graph.n_nodes)):
            raise ValueError("observation cell is outside the graph")
        if not np.all(np.isfinite(means)):
            raise ValueError("observation means must be finite")
        if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
            raise ValueError("observation variance must be finite and positive")
        np.add.at(self.local_precision, indices, 1.0 / variances)
        np.add.at(self.local_information, indices, means / variances)

    def observe_canonical(
        self,
        cells: np.ndarray,
        information: np.ndarray,
        precision: np.ndarray,
    ) -> None:
        precisions = np.asarray(precision, dtype=np.float64)
        informations = np.asarray(information, dtype=np.float64)
        if np.any(precisions <= 0.0):
            raise ValueError("canonical precision must be positive")
        self.observe(cells, informations / precisions, 1.0 / precisions)

    def belief(self) -> GaussianBelief:
        precision = self.local_precision.copy()
        information = self.local_information.copy()
        np.add.at(precision, self.target, self.message_precision)
        np.add.at(information, self.target, self.message_information)
        return GaussianBelief(information / precision, 1.0 / precision)

    def sweep(self, active_messages: np.ndarray | None = None) -> float:
        """Update messages once; a mask models loss or asynchronous scheduling."""

        if active_messages is None:
            active = np.ones(self.source.size, dtype=bool)
        else:
            active = np.asarray(active_messages, dtype=bool)
            if active.shape != self.source.shape:
                raise ValueError("active_messages must have one flag per message")
        if not np.any(active):
            return 0.0

        total_precision = self.local_precision.copy()
        total_information = self.local_information.copy()
        np.add.at(total_precision, self.target, self.message_precision)
        np.add.at(total_information, self.target, self.message_information)
        cavity_precision = (
            total_precision[self.source] - self.message_precision[self.reverse]
        )
        cavity_information = (
            total_information[self.source] - self.message_information[self.reverse]
        )
        outgoing_precision = 1.0 / (1.0 / cavity_precision + self.edge_variance)
        outgoing_information = (
            cavity_information / cavity_precision * outgoing_precision
        )

        damping = self.config.damping
        next_precision = self.message_precision.copy()
        next_information = self.message_information.copy()
        next_precision[active] += damping * (
            outgoing_precision[active] - next_precision[active]
        )
        next_information[active] += damping * (
            outgoing_information[active] - next_information[active]
        )
        residual = max(
            float(np.max(np.abs(next_precision - self.message_precision))),
            float(np.max(np.abs(next_information - self.message_information))),
        )
        self.message_precision[:] = next_precision
        self.message_information[:] = next_information
        return residual

    def run(self) -> GBPResult:
        residual = math.inf
        for sweep in range(1, self.config.max_sweeps + 1):
            residual = self.sweep()
            if residual <= self.config.tolerance:
                return GBPResult(self.belief(), sweep, residual, True)
        return GBPResult(self.belief(), self.config.max_sweeps, residual, False)

    def age(self, cells: np.ndarray, process_variance: float) -> None:
        """C4 time factor: stale local evidence loses precision smoothly."""

        if process_variance < 0.0 or not math.isfinite(process_variance):
            raise ValueError("process_variance must be finite and nonnegative")
        indices = np.asarray(cells)
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("aged cells must be integers")
        if np.any((indices < 0) | (indices >= self.graph.n_nodes)):
            raise ValueError("aged cell is outside the graph")
        mean = self.local_information[indices] / self.local_precision[indices]
        variance = 1.0 / self.local_precision[indices] + process_variance
        self.local_precision[indices] = 1.0 / variance
        self.local_information[indices] = mean / variance

    def age_messages(
        self,
        messages: np.ndarray,
        process_variance: float,
    ) -> None:
        """C4 time factor: unrefreshed edge messages lose precision."""

        if process_variance < 0.0 or not math.isfinite(process_variance):
            raise ValueError("process_variance must be finite and nonnegative")
        selected = np.asarray(messages)
        if selected.dtype == np.bool_:
            if selected.shape != self.source.shape:
                raise ValueError("message mask must have one flag per message")
            indices = np.flatnonzero(selected)
        else:
            if not np.issubdtype(selected.dtype, np.integer):
                raise ValueError("message indices must be integers or a mask")
            indices = selected.astype(np.int64, copy=False)
            if np.any((indices < 0) | (indices >= self.source.size)):
                raise ValueError("aged message is outside the graph")
        positive = indices[self.message_precision[indices] > 0.0]
        if positive.size == 0:
            return
        mean = (
            self.message_information[positive]
            / self.message_precision[positive]
        )
        variance = (
            1.0 / self.message_precision[positive] + process_variance
        )
        self.message_precision[positive] = 1.0 / variance
        self.message_information[positive] = mean / variance


__all__ = ["GBPConfig", "GBPResult", "GaussianBeliefPropagation"]
