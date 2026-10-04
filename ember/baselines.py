"""Comparable shortest-path and flooding baselines for Phase-1 missions."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from cascade.extract import TraversabilityGraph
from ember.comms import GaussianPacket
from mcu import Q16_SCALE


@dataclass(frozen=True, slots=True)
class BaselineResult:
    name: str
    route: tuple[int, ...]
    path_length: float
    hazard_dose: float
    radio_bytes: int

    @property
    def objective_cost(self) -> float:
        """Observer-only path objective used for clairvoyant regret."""

        return self.path_length + self.hazard_dose


def last_received_mean_map(
    n_nodes: int,
    packets: tuple[GaussianPacket, ...],
    *,
    prior_mean: float,
) -> np.ndarray:
    """Build B2's last-received map without invoking EMBER inference.

    Packets are applied in delivery order. A later observation of a cell
    replaces the older value, matching the deliberately simple flooding
    baseline contract.
    """

    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int) or n_nodes < 1:
        raise ValueError("n_nodes must be a positive integer")
    if not math.isfinite(prior_mean) or prior_mean < 0.0:
        raise ValueError("prior_mean must be finite and nonnegative")
    mean = np.full(n_nodes, prior_mean, dtype=np.float64)
    for packet in packets:
        cells = packet.cells.astype(np.int64)
        if np.any(cells >= n_nodes):
            raise ValueError("packet cell is outside the map")
        precision = packet.precision_q16.astype(np.float64) / Q16_SCALE
        information = packet.information_q16.astype(np.float64) / Q16_SCALE
        mean[cells] = information / precision
    return mean


def dijkstra_route(
    graph: TraversabilityGraph,
    start: int,
    goal: int,
    *,
    state_cost: np.ndarray | None = None,
) -> tuple[int, ...]:
    """Return the minimum edge-plus-state-cost route on an undirected graph."""

    if not 0 <= start < graph.n_nodes or not 0 <= goal < graph.n_nodes:
        raise ValueError("start and goal must be graph nodes")
    node_cost = (
        np.zeros(graph.n_nodes, dtype=np.float64)
        if state_cost is None
        else np.asarray(state_cost, dtype=np.float64)
    )
    if node_cost.shape != (graph.n_nodes,):
        raise ValueError("state_cost must have one entry per graph node")
    if np.any(node_cost < 0.0) or not np.all(np.isfinite(node_cost)):
        raise ValueError("state_cost must be finite and nonnegative")

    adjacency = graph.adjacency()
    distance = np.full(graph.n_nodes, np.inf)
    previous = np.full(graph.n_nodes, -1, dtype=np.int64)
    distance[start] = 0.0
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance != distance[node]:
            continue
        if node == goal:
            break
        for neighbour, edge_cost in adjacency[node]:
            candidate = current_distance + edge_cost + float(node_cost[neighbour])
            if candidate < distance[neighbour]:
                distance[neighbour] = candidate
                previous[neighbour] = node
                heapq.heappush(queue, (candidate, neighbour))
    if not math.isfinite(float(distance[goal])):
        raise ValueError("goal is unreachable from start")
    route = [goal]
    while route[-1] != start:
        route.append(int(previous[route[-1]]))
    route.reverse()
    return tuple(route)


def evaluate_route(
    name: str,
    graph: TraversabilityGraph,
    route: tuple[int, ...],
    truth_cost: np.ndarray,
    *,
    radio_bytes: int = 0,
) -> BaselineResult:
    if not route:
        raise ValueError("route cannot be empty")
    truth = np.asarray(truth_cost, dtype=np.float64)
    if truth.shape != (graph.n_nodes,):
        raise ValueError("truth_cost must have one entry per graph node")
    edge_lookup = {
        (min(int(a), int(b)), max(int(a), int(b))): float(cost)
        for (a, b), cost in zip(graph.edges, graph.edge_cost, strict=True)
    }
    length = 0.0
    for source, target in pairwise(route):
        key = (min(source, target), max(source, target))
        if key not in edge_lookup:
            raise ValueError("route contains a non-edge transition")
        length += edge_lookup[key]
    return BaselineResult(
        name=name,
        route=route,
        path_length=length,
        hazard_dose=float(np.sum(truth[list(route)])),
        radio_bytes=radio_bytes,
    )


def centralized_shortest_baseline(
    graph: TraversabilityGraph,
    start: int,
    goal: int,
    truth_cost: np.ndarray,
) -> BaselineResult:
    route = dijkstra_route(graph, start, goal)
    return evaluate_route("B1 centralized shortest", graph, route, truth_cost)


def flooding_dijkstra_baseline(
    graph: TraversabilityGraph,
    start: int,
    goal: int,
    shared_cost: np.ndarray,
    truth_cost: np.ndarray,
    *,
    radio_bytes: int,
) -> BaselineResult:
    route = dijkstra_route(graph, start, goal, state_cost=shared_cost)
    return evaluate_route(
        "B2 flooding Dijkstra",
        graph,
        route,
        truth_cost,
        radio_bytes=radio_bytes,
    )


def clairvoyant_dijkstra_baseline(
    graph: TraversabilityGraph,
    start: int,
    goal: int,
    truth_cost: np.ndarray,
) -> BaselineResult:
    """Solve the edge-plus-hazard objective using held-out simulator truth.

    This is an evaluation oracle, never an input to a robot. Its objective is
    identical to :func:`dijkstra_route`: edge length plus the entered cells'
    state costs. The start-cell term is constant across policies, so including
    it when reporting ``objective_cost`` does not change regret.
    """

    route = dijkstra_route(graph, start, goal, state_cost=truth_cost)
    return evaluate_route("clairvoyant Dijkstra", graph, route, truth_cost)


__all__ = [
    "BaselineResult",
    "centralized_shortest_baseline",
    "clairvoyant_dijkstra_baseline",
    "dijkstra_route",
    "evaluate_route",
    "flooding_dijkstra_baseline",
    "last_received_mean_map",
]
