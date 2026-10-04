"""Procedural survivor placement and cooperative BLE search decisions.

Ground truth stays in the simulator.  Robot-side targets come only from noisy
RSSI observations fused after the swarm has spread to a useful baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cascade.extract import TraversabilityGraph, dijkstra_from
from ember.localization import RSSIConfig, localize_rssi, sample_rssi


@dataclass(frozen=True, slots=True)
class CooperativeSearchResult:
    truth_positions_m: np.ndarray
    estimated_positions_m: np.ndarray
    covariance_m2: np.ndarray
    rssi_dbm: np.ndarray
    robot_to_beacon: tuple[int, ...]

    @property
    def errors_m(self) -> np.ndarray:
        return np.linalg.norm(
            self.estimated_positions_m - self.truth_positions_m, axis=1
        )

    @property
    def max_error_m(self) -> float:
        return float(np.max(self.errors_m))

    @property
    def uncertainty_radii_m(self) -> np.ndarray:
        """Major-axis radii of the fused 95% confidence ellipses."""

        largest = np.linalg.eigvalsh(self.covariance_m2)[:, -1]
        return np.sqrt(5.991 * np.maximum(largest, 0.0))


def sample_survivor_sites(
    graph: TraversabilityGraph,
    origin: int,
    count: int,
    footprint: tuple[float, float, float, float],
    rng: np.random.Generator,
    *,
    minimum_depth_m: float = 2.0,
    minimum_route_cost: float = 12.0,
    maximum_route_cost: float | None = None,
) -> np.ndarray:
    """Place hidden survivors at reachable, deep, well-separated floor cells."""

    if count < 1:
        raise ValueError("survivor count must be positive")
    if not 0 <= origin < graph.n_nodes:
        raise ValueError("origin must be a graph node")
    x0, y0, x1, y1 = footprint
    positions = graph.world_positions()
    depth = np.minimum.reduce(
        (
            positions[:, 0] - x0,
            x1 - positions[:, 0],
            positions[:, 1] - y0,
            y1 - positions[:, 1],
        )
    )
    distance, _ = dijkstra_from(graph, origin)
    candidates = np.flatnonzero(
        np.isfinite(distance)
        & (distance >= minimum_route_cost)
        & (True if maximum_route_cost is None else distance <= maximum_route_cost)
        & (depth >= minimum_depth_m)
        & np.isclose(positions[:, 2], positions[origin, 2])
    )
    if candidates.size < count:
        raise RuntimeError("not enough reachable interior cells for survivors")

    # Seeded first choice prevents every world from placing survivor zero in
    # the same geometric corner.  Subsequent sites maximize separation, which
    # makes independent search assignments visible and meaningful.
    far = candidates[distance[candidates] >= np.quantile(distance[candidates], 0.7)]
    chosen = [int(far[int(rng.integers(len(far)))])]
    while len(chosen) < count:
        separation = np.min(
            np.linalg.norm(
                positions[candidates, :2][:, None, :]
                - positions[chosen, :2][None, :, :],
                axis=2,
            ),
            axis=1,
        )
        separation[np.isin(candidates, chosen)] = -1.0
        chosen.append(int(candidates[int(np.argmax(separation))]))
    return positions[chosen]


def localize_and_assign(
    robot_positions_m: np.ndarray,
    survivor_positions_m: np.ndarray,
    rng: np.random.Generator,
    *,
    config: RSSIConfig | None = None,
) -> CooperativeSearchResult:
    """Fuse BLE ranges, then compute the leaderless common minimum matching."""

    robots = np.asarray(robot_positions_m, dtype=np.float64)
    survivors = np.asarray(survivor_positions_m, dtype=np.float64)
    if robots.ndim != 2 or robots.shape[1] != 2 or robots.shape[0] < 3:
        raise ValueError("at least three 2-D robot positions are required")
    if survivors.shape != (robots.shape[0], 2):
        raise ValueError("this search slice requires one survivor per robot")
    cfg = config or RSSIConfig(noise_std_db=0.45)
    estimates = []
    covariances = []
    readings = []
    for survivor in survivors:
        rssi = sample_rssi(robots, survivor, rng, config=cfg)
        readings.append(rssi)
        estimate = localize_rssi(robots, rssi, config=cfg)
        estimates.append(estimate.position_m)
        covariances.append(estimate.covariance_m2)
    estimated = np.asarray(estimates)
    cost = np.linalg.norm(robots[:, None, :] - estimated[None, :, :], axis=2)
    # Every robot observes the same fused table and applies the same stable
    # ordering, so this needs neither a leader nor negotiation.  Greedy common
    # matching remains cheap at the proposal's 8--24 robot scale, unlike an
    # exhaustive permutation search.
    assignments = np.full(len(robots), -1, dtype=np.int64)
    claimed: set[int] = set()
    for flat_index in np.argsort(cost, axis=None, kind="stable"):
        robot, beacon = np.unravel_index(flat_index, cost.shape)
        if assignments[robot] >= 0 or int(beacon) in claimed:
            continue
        assignments[robot] = beacon
        claimed.add(int(beacon))
        if len(claimed) == len(survivors):
            break
    return CooperativeSearchResult(
        truth_positions_m=survivors,
        estimated_positions_m=estimated,
        covariance_m2=np.asarray(covariances),
        rssi_dbm=np.asarray(readings).T,
        robot_to_beacon=tuple(int(value) for value in assignments),
    )


__all__ = [
    "CooperativeSearchResult",
    "localize_and_assign",
    "sample_survivor_sites",
]
