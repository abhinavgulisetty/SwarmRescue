"""Procedural survivor placement and cooperative BLE search decisions.

Ground truth stays in the simulator.  Robot-side targets come only from noisy
RSSI observations fused after the swarm has spread to a useful baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cascade.extract import TraversabilityGraph, dijkstra_from
from ember.localization import RSSIConfig, localize_rssi, sample_rssi
from ember.radio import RadioMedium, RadioStats
from ember.range_gbp import RangeFactorGBP
from ember.range_team import RangeLocalizationTeam


@dataclass(frozen=True, slots=True)
class CooperativeSearchResult:
    truth_positions_m: np.ndarray
    estimated_positions_m: np.ndarray
    covariance_m2: np.ndarray
    rssi_dbm: np.ndarray
    robot_to_beacon: tuple[int, ...]
    range_gbp: bool = False
    range_gbp_converged: bool | None = None
    range_gbp_sweeps: int = 0
    range_gbp_linearizations: int = 0
    localization_retained_numeric_bytes: int = 0
    range_radio_enabled: bool = False
    range_radio_rounds: int = 0
    range_radio_stats: RadioStats = field(default_factory=RadioStats)
    range_receiver_factor_counts: tuple[int, ...] = ()
    localization_team_numeric_bytes: int = 0
    receiver_estimated_positions_m: np.ndarray | None = None
    receiver_covariance_m2: np.ndarray | None = None

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
    range_gbp: bool = False,
    pose_covariance_m2: np.ndarray | None = None,
    range_radio: RadioMedium | None = None,
    range_max_rounds: int = 3,
) -> CooperativeSearchResult:
    """Infer receiver targets, then coordinate a common one-to-one assignment.

    Radio mode supplies only each receiver's delivered factors. The simulation
    host still collects cost bids and commits matching; bid transport and
    independent agreement under partitions are not modeled here.
    """
    robots = np.asarray(robot_positions_m, dtype=np.float64)
    survivors = np.asarray(survivor_positions_m, dtype=np.float64)
    if robots.ndim != 2 or robots.shape[1] != 2 or robots.shape[0] < 3:
        raise ValueError("at least three 2-D robot positions are required")
    if survivors.shape != (robots.shape[0], 2):
        raise ValueError("this search slice requires one survivor per robot")
    if range_radio is not None and not range_gbp:
        raise ValueError("range radio requires range GBP")
    if pose_covariance_m2 is not None and not range_gbp:
        raise ValueError("pose covariance requires range GBP")
    cfg = config or RSSIConfig(noise_std_db=0.45)
    engine, team = None, None
    if range_gbp:
        covariance = (
            np.tile(np.eye(2) * 1e-6, (len(robots), 1, 1))
            if pose_covariance_m2 is None
            else pose_covariance_m2
        )
        if range_radio is None:
            engine = RangeFactorGBP(robots, covariance, len(survivors))
        else:
            team = RangeLocalizationTeam(
                robots, covariance, len(survivors), range_radio, sensor_config=cfg
            )
    estimates, covariances, readings = [], [], []
    for beacon, survivor in enumerate(survivors):
        rssi = sample_rssi(robots, survivor, rng, config=cfg)
        readings.append(rssi)
        if engine is not None:
            for source, reading in enumerate(rssi):
                engine.observe_rssi(source, beacon, 0, float(reading), config=cfg)
        elif team is None:
            estimate = localize_rssi(robots, rssi, config=cfg)
            estimates.append(estimate.position_m)
            covariances.append(estimate.covariance_m2)
    solution = engine.solve() if engine is not None else None
    if solution is not None:
        if not solution.converged:
            raise RuntimeError("coupled range GBP did not converge")
        estimates = [estimate.position_m for estimate in solution.estimates]
        covariances = [estimate.covariance_m2 for estimate in solution.estimates]
    transport = (
        team.exchange(np.asarray(readings).T, rng, max_rounds=range_max_rounds)
        if team
        else None
    )
    receiver_positions, receiver_covariances = None, None
    if transport:
        if not transport.available:
            unavailable = [
                str(i)
                for i, result in enumerate(transport.solutions)
                if result is None or not result.converged
            ]
            raise RuntimeError(
                "range localization unavailable for receivers " + ", ".join(unavailable)
            )
        receiver_positions = np.asarray(
            [
                [estimate.position_m for estimate in result.estimates]
                for result in transport.solutions
            ]
        )
        receiver_covariances = np.asarray(
            [
                [estimate.covariance_m2 for estimate in result.estimates]
                for result in transport.solutions
            ]
        )
        cost = np.linalg.norm(robots[:, None, :] - receiver_positions, axis=2)
    else:
        estimated = np.asarray(estimates)
        cost = np.linalg.norm(robots[:, None, :] - estimated[None, :, :], axis=2)
    # This host coordinator uses local cost rows in radio mode. No shared
    # posterior or averaged peer estimate enters an assigned robot's target.
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
    if transport:
        estimated = np.empty_like(survivors)
        covariances = np.empty((len(survivors), 2, 2))
        for receiver, beacon in enumerate(assignments):
            estimated[beacon] = receiver_positions[receiver, beacon]
            covariances[beacon] = receiver_covariances[receiver, beacon]
    return CooperativeSearchResult(
        range_gbp=range_gbp,
        range_gbp_converged=True
        if transport
        else solution.converged
        if solution
        else None,
        range_gbp_sweeps=sum(r.sweeps for r in transport.solutions)
        if transport
        else solution.sweeps
        if solution
        else 0,
        range_gbp_linearizations=max(r.linearizations for r in transport.solutions)
        if transport
        else solution.linearizations
        if solution
        else 0,
        localization_retained_numeric_bytes=transport.receiver_numeric_bytes
        if transport
        else solution.retained_numeric_bytes
        if solution
        else 0,
        localization_team_numeric_bytes=transport.team_numeric_bytes
        if transport
        else solution.retained_numeric_bytes
        if solution
        else 0,
        range_radio_enabled=transport is not None,
        range_radio_rounds=transport.rounds if transport else 0,
        range_radio_stats=transport.radio_stats if transport else RadioStats(),
        range_receiver_factor_counts=transport.factor_counts if transport else (),
        receiver_estimated_positions_m=receiver_positions,
        receiver_covariance_m2=receiver_covariances,
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
