"""Complete-rescue bids from private fields and an explicit timing model.

Linear assignment and Dijkstra routing are established methods. Certificates
cover the supplied frozen belief-cost model, not unobserved physical fire.
Assignment remains host coordinated, and physical execution can invalidate its
route/timing assumptions. No calibrated survival probability is implied.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ember.rescue_certificate import certify_rescue


@dataclass(frozen=True)
class RescueBid:
    robot: int
    survivor: int
    approach: tuple[int, ...]
    escort: tuple[int, ...]
    robot_bound: float
    survivor_bound: float
    latest_completion_s: float
    objective: float
    feasible: bool


@dataclass(frozen=True)
class RescueAssignment:
    robot_to_survivor: tuple[int, ...]
    selected: tuple[RescueBid | None, ...]
    bids: tuple[RescueBid, ...]
    rejected_pairs: int
    method: str = "complete-rescue-dijkstra-linear-assignment"
    certificate_model: str = "private-frozen-routing-cost"


def private_fine_risk(team, *, beta=2.0):
    """Read each robot's own field, including its direct fine observations."""
    rows = []
    for robot in team.robots:
        risk = robot.engine.belief().routing_cost(beta)
        if hasattr(team, "regions"):
            risk = risk[team.regions.node_to_region].copy()
            observed = robot.fine_observed
            risk[observed] = np.maximum(
                robot.fine_mean[observed].astype(float)
                + beta * np.sqrt(robot.fine_variance[observed].astype(float)),
                1e-4,
            )
        rows.append(risk)
    return np.asarray(rows)


def _paths(adjacency, roots):
    distance = np.full(len(adjacency), np.inf)
    parent = np.full(len(adjacency), -1, dtype=int)
    queue = []
    for root in sorted(set(roots)):
        distance[root] = 0.0
        heapq.heappush(queue, (0.0, root))
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance[node]:
            continue
        for target, weight in adjacency[node]:
            candidate = cost + weight
            if candidate < distance[target]:
                distance[target], parent[target] = candidate, node
                heapq.heappush(queue, (candidate, target))
    return distance, parent


def _to_root(node, parent):
    path = [int(node)]
    while parent[path[-1]] >= 0:
        path.append(int(parent[path[-1]]))
    return tuple(path)


def complete_rescue_assignment(
    positions,
    edges,
    starts,
    targets,
    exits,
    private_risk,
    *,
    available_time_s,
    approach_speed_m_s=0.35,
    escort_speed_m_s=0.25,
    dt_s=0.5,
    lag_fraction=0.25,
    robot_budget=np.inf,
    survivor_budget=np.inf,
):
    """Max-cardinality matching of complete, model-feasible candidate rescues.

    ``targets[r,v]`` is receiver r's node estimate of survivor v. Each robot
    supplies its own risk row; no averaged team posterior is constructed.
    Candidates combine risk-weighted and physical-distance approach/exit paths.
    Every retained alternative receives its own complete-rescue certificate;
    the cheapest feasible alternative supplies each robot-survivor matching bid.
    The certificate includes victim waiting and every declared escort dwell.
    Global optimality is only over these candidate bids, not all graph routes.
    """
    positions, private_risk = np.asarray(positions, float), np.asarray(private_risk)
    starts, targets, exits = map(np.asarray, (starts, targets, exits))
    edges = np.asarray(edges)
    n = len(positions)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or not np.all(np.isfinite(positions))
        or starts.ndim != 1
        or targets.ndim != 2
        or targets.shape[0] != len(starts)
        or private_risk.shape != (len(starts), n)
        or exits.ndim != 1
        or edges.ndim != 2
        or edges.shape[1] != 2
    ):
        raise ValueError("incompatible complete-rescue graph/actor dimensions")
    for indices in (starts, targets, exits, edges):
        if not np.issubdtype(indices.dtype, np.integer) or np.any(
            (indices < 0) | (indices >= n)
        ):
            raise ValueError("graph indices must be in-range integers")
    if (
        not np.all(np.isfinite(private_risk))
        or np.any(private_risk < 0)
        or not np.isfinite(available_time_s)
        or available_time_s <= 0
        or not np.isfinite(lag_fraction)
        or lag_fraction < 0
        or any(
            not np.isfinite(value) or value <= 0
            for value in (dt_s, approach_speed_m_s, escort_speed_m_s)
        )
    ):
        raise ValueError("risk, deadline and timing parameters are invalid")
    if any(np.isnan(value) or value < 0 for value in (robot_budget, survivor_budget)):
        raise ValueError("exposure budgets must be nonnegative")
    if not len(starts) or not targets.shape[1]:
        return RescueAssignment(
            tuple(-1 for _ in starts), tuple(None for _ in starts), (), 0
        )
    lengths = np.linalg.norm(positions[edges[:, 0]] - positions[edges[:, 1]], axis=1)
    if np.any(lengths <= 0):
        raise ValueError("graph moves must have positive physical lengths")
    geometric = [[] for _ in range(n)]
    for (left, right), length in zip(edges, lengths, strict=True):
        geometric[left].append((int(right), float(length)))
        geometric[right].append((int(left), float(length)))
    distance_outbound, distance_escort_parent = _paths(geometric, map(int, exits))

    def ticks(path, speed):
        length = np.linalg.norm(np.diff(positions[list(path)], axis=0), axis=1)
        return np.maximum(1, np.ceil(length / speed / dt_s)).astype(int)

    bids = []
    for robot, risk in enumerate(private_risk):
        adjacency = [[] for _ in range(n)]
        for (left, right), length in zip(edges, lengths, strict=True):
            weight = float(length * (0.01 + 0.5 * (risk[left] + risk[right])))
            adjacency[left].append((int(right), weight))
            adjacency[right].append((int(left), weight))
        inbound, approach_parent = _paths(adjacency, (int(starts[robot]),))
        outbound, escort_parent = _paths(adjacency, map(int, exits))
        distance_inbound, distance_approach_parent = _paths(
            geometric, (int(starts[robot]),)
        )
        for survivor, target in enumerate(targets[robot]):
            approaches = tuple(
                dict.fromkeys(
                    _to_root(target, parent)[::-1]
                    for distance, parent in (
                        (inbound, approach_parent),
                        (distance_inbound, distance_approach_parent),
                    )
                    if np.isfinite(distance[target])
                )
            )
            escorts = tuple(
                dict.fromkeys(
                    _to_root(target, parent)
                    for distance, parent in (
                        (outbound, escort_parent),
                        (distance_outbound, distance_escort_parent),
                    )
                    if np.isfinite(distance[target])
                )
            )
            if not approaches or not escorts:
                continue
            for approach in approaches:
                for escort in escorts:
                    approach_ticks = ticks(approach, approach_speed_m_s)
                    escort_min = ticks(escort, escort_speed_m_s)
                    escort_max = np.ceil(escort_min * (1 + lag_fraction)).astype(int)
                    completion = int(approach_ticks.sum() + escort_max.sum())
                    # A failed necessary deadline test rejects the alternative;
                    # it never becomes an uncertified geometric fallback.
                    if completion * dt_s > available_time_s:
                        continue
                    used = np.unique((*approach, *escort))
                    mapping = {int(node): i for i, node in enumerate(used)}
                    field = np.broadcast_to(
                        risk[used], (1, max(1, completion), len(used))
                    )
                    certificate = certify_rescue(
                        field,
                        tuple(mapping[node] for node in approach),
                        tuple(mapping[node] for node in escort),
                        approach_ticks=approach_ticks,
                        escort_min_ticks=escort_min,
                        escort_max_ticks=escort_max,
                        robot_budget=robot_budget,
                        survivor_budget=survivor_budget,
                        dt=dt_s,
                    )
                    objective = (
                        certificate.survivor_bound
                        + 0.1 * certificate.robot_bound
                        + 0.01 * completion * dt_s
                    )
                    bids.append(
                        RescueBid(
                            robot,
                            survivor,
                            approach,
                            escort,
                            certificate.robot_bound,
                            certificate.survivor_bound,
                            completion * dt_s,
                            objective,
                            certificate.feasible,
                        )
                    )
    feasible = {}
    for bid in bids:
        key = bid.robot, bid.survivor
        if bid.feasible and (
            key not in feasible or bid.objective < feasible[key].objective
        ):
            feasible[key] = bid
    maximum = max((bid.objective for bid in feasible.values()), default=0.0)
    # One dummy per robot permits idle robots and impossible/unequal task sets.
    # This penalty makes cardinality lexicographically precede nonnegative cost.
    penalty = (len(starts) + 1) * (maximum + 1)
    costs = np.full((len(starts), targets.shape[1] + len(starts)), penalty)
    costs[:, : targets.shape[1]] = penalty * (len(starts) + 1)
    for key, bid in feasible.items():
        costs[key] = bid.objective
    row_indices, columns = linear_sum_assignment(costs)
    assignments, selected = [-1] * len(starts), [None] * len(starts)
    for row, column in zip(row_indices, columns, strict=True):
        if (int(row), int(column)) in feasible:
            assignments[row] = int(column)
            selected[row] = feasible[int(row), int(column)]
    return RescueAssignment(
        tuple(assignments),
        tuple(selected),
        tuple(bids),
        int(len(starts) * targets.shape[1] - len(feasible)),
    )
