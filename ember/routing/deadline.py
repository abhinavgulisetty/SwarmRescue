"""Integer Pareto routing with a hard evacuation-time constraint.

Each label keeps both elapsed discrete traversal steps and estimated risk cost.
Dominance preserves a faster alternative when the lowest-risk path is too slow.
The graph, risk estimates, speed and deadline are policy inputs; future hazard
truth is never read. A bounded label arena falls back to the fastest route when
full, retaining feasibility rather than pretending optimality.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np

from cascade.extract import TraversabilityGraph
from mcu import INT32_MAX, Q16_SCALE, MCUArena


@dataclass(frozen=True, slots=True)
class DeadlineRoute:
    route: tuple[int, ...]
    travel_steps: int
    estimated_objective: float
    feasible: bool
    optimal: bool
    resource_fallback: bool
    labels_used: int
    arena_high_water_bytes: int
    proposal_replaced: bool = False
    lower_bound_q16: int | None = None
    search_strategy: str = "pareto"
    expanded_labels: int = 0
    preprocessing_runs: int = 0
    dynamic_states: int = 0

    @property
    def certified_gap(self) -> float | None:
        """Absolute gap in the declared estimated objective, not true exposure."""
        if not self.feasible or self.lower_bound_q16 is None:
            return None
        return max(0.0, self.estimated_objective - self.lower_bound_q16 / Q16_SCALE)


def deadline_route(
    graph: TraversabilityGraph,
    start: int,
    goals: tuple[int, ...],
    risk_cost: np.ndarray,
    *,
    available_steps: int,
    walking_speed_m_s: float,
    dt_s: float,
    max_labels: int = 4096,
    arena: MCUArena | None = None,
    strategy: str = "pareto",
) -> DeadlineRoute:
    """Minimize length plus estimated exposure subject to a traversal deadline."""
    if strategy not in (
        "pareto",
        "astar",
        "lazy_astar",
        "dual_astar",
        "hybrid_exact",
        "bounded_dp",
        "window_dp",
    ):
        raise ValueError("unknown deadline search strategy")
    risk = np.asarray(risk_cost, dtype=float)
    if (
        risk.shape != (graph.n_nodes,)
        or np.any(risk < 0)
        or not np.all(np.isfinite(risk))
    ):
        raise ValueError("risk must be finite and nonnegative with one value per node")
    if (
        not 0 <= start < graph.n_nodes
        or not goals
        or any(not 0 <= goal < graph.n_nodes for goal in goals)
    ):
        raise ValueError("route endpoints must be graph nodes")
    if (
        not isinstance(available_steps, int)
        or isinstance(available_steps, bool)
        or available_steps < 0
        or available_steps >= INT32_MAX
    ):
        raise ValueError("available_steps must fit the nonnegative int32 clock")
    if (
        not isinstance(max_labels, int)
        or isinstance(max_labels, bool)
        or max_labels < 1
        or min(walking_speed_m_s, dt_s) <= 0
        or not all(math.isfinite(value) for value in (walking_speed_m_s, dt_s))
    ):
        raise ValueError("speed, time step and label capacity must be positive")
    if (
        graph.edges.ndim != 2
        or graph.edges.shape[1] != 2
        or not np.issubdtype(graph.edges.dtype, np.integer)
        or graph.edge_cost.shape != (graph.n_edges,)
        or np.any(graph.edges < 0)
        or np.any(graph.edges >= graph.n_nodes)
        or not np.all(np.isfinite(graph.edge_cost))
        or np.any(graph.edge_cost < 0)
    ):
        raise ValueError(
            "graph edges require valid endpoints and finite nonnegative lengths"
        )
    adjacent = [[] for _ in range(graph.n_nodes)]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        travel = np.maximum(1.0, np.ceil(graph.edge_cost / walking_speed_m_s / dt_s))
        if not np.all(np.isfinite(travel)) or np.any(travel >= INT32_MAX):
            raise ValueError("edge traversal time exceeds int32 clock range")
        steps = travel.astype(np.int64)
        encoded = np.rint(
            (graph.edge_cost[:, None] + risk[graph.edges] * steps[:, None] * dt_s)
            * Q16_SCALE
        )
        if not np.all(np.isfinite(encoded)) or np.any(encoded >= INT32_MAX):
            raise ValueError("edge objective exceeds integer finite range")
        costs = np.maximum(1, encoded).astype(np.int64)
    for index, (left, right) in enumerate(graph.edges):
        a, b, duration = int(left), int(right), int(steps[index])
        adjacent[a].append((b, duration, int(costs[index, 1])))
        adjacent[b].append((a, duration, int(costs[index, 0])))
    goal_set = set(goals)
    if strategy != "pareto":
        from ember.routing.constrained import constrained_search

        return constrained_search(
            adjacent,
            start,
            goals,
            available_steps,
            max_labels,
            arena=arena,
            strategy=strategy,
            legacy_fallback=lambda: deadline_route(
                graph,
                start,
                goals,
                risk_cost,
                available_steps=available_steps,
                walking_speed_m_s=walking_speed_m_s,
                dt_s=dt_s,
                max_labels=max_labels,
                arena=arena,
            ),
        )

    # A lexicographic fastest-path certificate is also the bounded-memory fallback.
    fastest = {start: (0, 0)}
    predecessor = {}
    queue = [(0, 0, start)]
    fastest_goal = -1
    while queue:
        elapsed, cost, node = heapq.heappop(queue)
        if fastest[node] != (elapsed, cost):
            continue
        if node in goal_set:
            fastest_goal = node
            break
        for target, steps, edge_cost in adjacent[node]:
            candidate = (elapsed + steps, cost + edge_cost)
            if candidate < fastest.get(target, (INT32_MAX, INT32_MAX)):
                fastest[target] = candidate
                predecessor[target] = node
                heapq.heappush(queue, (*candidate, target))
    fast_route = []
    if fastest_goal >= 0:
        fast_route = [fastest_goal]
        while fast_route[-1] != start:
            fast_route.append(predecessor[fast_route[-1]])
        fast_route.reverse()
    fast_time, fast_cost = fastest.get(fastest_goal, (0, 0))
    if fastest_goal < 0 or fast_time > available_steps:
        return DeadlineRoute(
            tuple(fast_route),
            fast_time,
            fast_cost / Q16_SCALE,
            False,
            True,
            False,
            0,
            0,
        )

    storage = arena if arena is not None else MCUArena(200 * 1024)
    offset = storage.used
    aligned = (offset + 3) & ~3
    # Four int32 label fields plus a live byte. Never expand a caller's arena.
    max_labels = min(max_labels, max(0, (storage.capacity - aligned) // 17))
    if max_labels == 0:
        return DeadlineRoute(
            tuple(fast_route),
            fast_time,
            fast_cost / Q16_SCALE,
            True,
            False,
            True,
            0,
            storage.high_water,
        )
    try:
        labels = np.frombuffer(
            storage.alloc(max_labels * 4 * 4), dtype=np.int32
        ).reshape(-1, 4)
        live = np.frombuffer(storage.alloc(max_labels), dtype=np.uint8)
        live[:] = 0
        labels[0] = (start, 0, 0, -1)
        live[0] = 1
        per_node = [[] for _ in range(graph.n_nodes)]
        per_node[start].append(0)
        used = 1
        queue = [(0, 0, 0)]  # objective, elapsed, label id
        while queue:
            cost, elapsed, label_id = heapq.heappop(queue)
            if not live[label_id]:
                continue
            node = int(labels[label_id, 0])
            if node in goal_set:
                route = [node]
                previous = int(labels[label_id, 3])
                while previous >= 0:
                    route.append(int(labels[previous, 0]))
                    previous = int(labels[previous, 3])
                route.reverse()
                return DeadlineRoute(
                    tuple(route),
                    elapsed,
                    cost / Q16_SCALE,
                    True,
                    True,
                    False,
                    used,
                    storage.high_water,
                )
            for target, steps, edge_cost in adjacent[node]:
                next_time, next_cost = elapsed + steps, cost + edge_cost
                if next_time > available_steps or next_cost >= INT32_MAX:
                    continue
                frontier = per_node[target]
                if any(
                    live[index]
                    and labels[index, 1] <= next_time
                    and labels[index, 2] <= next_cost
                    for index in frontier
                ):
                    continue
                if used == max_labels:
                    return DeadlineRoute(
                        tuple(fast_route),
                        fast_time,
                        fast_cost / Q16_SCALE,
                        True,
                        False,
                        True,
                        used,
                        storage.high_water,
                    )
                for index in frontier:
                    if labels[index, 1] >= next_time and labels[index, 2] >= next_cost:
                        live[index] = 0
                labels[used] = (target, next_time, next_cost, label_id)
                live[used] = 1
                frontier.append(used)
                heapq.heappush(queue, (next_cost, next_time, used))
                used += 1
        # Rounding/saturation may make a finite path unrepresentable in labels.
        return DeadlineRoute(
            tuple(fast_route),
            fast_time,
            fast_cost / Q16_SCALE,
            True,
            False,
            True,
            used,
            storage.high_water,
        )

    finally:
        # Labels are scratch; preserve the policy and unrelated arena prefix.
        storage.used = offset


def safeguard_route_deadline(
    graph,
    start,
    goals,
    risk_cost,
    proposal,
    *,
    available_steps,
    walking_speed_m_s,
    dt_s,
    arena=None,
    strategy="hybrid_exact",
):
    """Preserve a feasible proposal; replace an infeasible one by exact hybrid routing.

    With common contact times, a static graph and the declared edge traversal
    clock, every feasible legacy rescue stays feasible. If any path meets the
    deadline, the fastest-path certificate/fallback also meets it. This is a
    modeled completion guarantee, not a tenability or physical timing proof.
    """
    lookup = {
        (min(int(left), int(right)), max(int(left), int(right))): float(length)
        for (left, right), length in zip(graph.edges, graph.edge_cost, strict=True)
    }
    risk = np.asarray(risk_cost)
    elapsed = cost = 0
    valid = bool(proposal) and proposal[0] == start and proposal[-1] in goals
    if valid:
        for left, right in pairwise(proposal):
            length = lookup.get((min(left, right), max(left, right)))
            if length is None:
                valid = False
                break
            steps = max(1, math.ceil(length / walking_speed_m_s / dt_s))
            elapsed += steps
            cost += round((length + risk[right] * steps * dt_s) * Q16_SCALE)
    if valid and elapsed <= available_steps:
        return DeadlineRoute(
            tuple(proposal),
            elapsed,
            cost / Q16_SCALE,
            True,
            False,
            False,
            0,
            0,
            False,
        )
    return replace(
        deadline_route(
            graph,
            start,
            goals,
            risk_cost,
            available_steps=available_steps,
            walking_speed_m_s=walking_speed_m_s,
            dt_s=dt_s,
            arena=arena,
            strategy=strategy,
        ),
        proposal_replaced=True,
    )
