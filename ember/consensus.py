"""Distributed Gaussian average-consensus baseline for B3."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ember.belief import GaussianBelief


@dataclass(frozen=True, slots=True)
class ConsensusConfig:
    tolerance: float = 1.0e-8
    max_rounds: int = 1_000

    def __post_init__(self) -> None:
        if self.tolerance <= 0.0 or not math.isfinite(self.tolerance):
            raise ValueError("consensus tolerance must be finite and positive")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    beliefs: tuple[GaussianBelief, ...]
    rounds: int
    residual: float
    converged: bool
    component_count: int


def _metropolis(adjacency: np.ndarray) -> np.ndarray:
    links = np.asarray(adjacency, dtype=bool)
    if links.ndim != 2 or links.shape[0] != links.shape[1] or links.shape[0] < 1:
        raise ValueError("adjacency must be a nonempty square matrix")
    if not np.array_equal(links, links.T):
        raise ValueError("consensus adjacency must be symmetric")
    links = links.copy()
    np.fill_diagonal(links, False)
    degree = np.sum(links, axis=1)
    weights = np.zeros(links.shape, dtype=np.float64)
    for left in range(links.shape[0]):
        for right in np.flatnonzero(links[left]):
            weights[left, right] = 1.0 / (
                1.0 + max(int(degree[left]), int(degree[right]))
            )
        weights[left, left] = 1.0 - float(np.sum(weights[left]))
    return weights


def _components(adjacency: np.ndarray) -> tuple[np.ndarray, ...]:
    links = np.asarray(adjacency, dtype=bool)
    unseen = set(range(links.shape[0]))
    groups = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        group = {root}
        stack = [root]
        while stack:
            node = stack.pop()
            for neighbour in np.flatnonzero(links[node]):
                item = int(neighbour)
                if item in unseen:
                    unseen.remove(item)
                    group.add(item)
                    stack.append(item)
        groups.append(np.asarray(sorted(group), dtype=np.int64))
    return tuple(groups)


def gaussian_average_consensus(
    beliefs: tuple[GaussianBelief, ...] | list[GaussianBelief],
    adjacency: np.ndarray,
    *,
    config: ConsensusConfig | None = None,
) -> ConsensusResult:
    """Reach component-wise consensus in canonical Gaussian parameters.

    Each connected component recovers the sum of its members' canonical evidence
    after average consensus. This deliberately naive B3 baseline assumes independent
    evidence and therefore exposes the data-incest weakness that EMBER avoids.
    """

    items = tuple(beliefs)
    if not items:
        raise ValueError("consensus needs at least one robot belief")
    cells = items[0].mean.size
    if any(item.mean.size != cells for item in items):
        raise ValueError("all robot beliefs must have the same cell count")
    weights = _metropolis(adjacency)
    if weights.shape[0] != len(items):
        raise ValueError("adjacency size must match the robot count")
    groups = _components(adjacency)
    precision = np.stack([1.0 / item.variance for item in items])
    information = np.stack(
        [item.mean / item.variance for item in items]
    )
    cfg = config or ConsensusConfig()
    residual = math.inf
    rounds = 0
    for round_index in range(1, cfg.max_rounds + 1):
        rounds = round_index
        next_precision = weights @ precision
        next_information = weights @ information
        residual = max(
            float(np.max(np.abs(next_precision - precision))),
            float(np.max(np.abs(next_information - information))),
        )
        precision = next_precision
        information = next_information
        if residual <= cfg.tolerance:
            break

    recovered_precision = np.empty_like(precision)
    recovered_information = np.empty_like(information)
    for group in groups:
        scale = float(group.size)
        recovered_precision[group] = precision[group] * scale
        recovered_information[group] = information[group] * scale
    result = tuple(
        GaussianBelief(
            recovered_information[index] / recovered_precision[index],
            1.0 / recovered_precision[index],
        )
        for index in range(len(items))
    )
    return ConsensusResult(
        beliefs=result,
        rounds=rounds,
        residual=residual,
        converged=residual <= cfg.tolerance,
        component_count=len(groups),
    )


__all__ = [
    "ConsensusConfig",
    "ConsensusResult",
    "gaussian_average_consensus",
]
