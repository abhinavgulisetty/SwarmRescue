"""Independent time-expanded optima and honest bounded-search certificates."""

import math
from itertools import pairwise

import numpy as np
import pytest

from cascade.extract import TraversabilityGraph, VoxelGrid
from ember.routing.deadline import deadline_route
from mcu import INT32_MAX, MCUArena

STRATEGIES = ("astar", "lazy_astar", "dual_astar")


def graph_from_edges(n, edges, lengths):
    occupied = np.zeros((n, 1, 1), bool)
    return TraversabilityGraph(
        VoxelGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), occupied, occupied),
        np.array([[i, 0, 0] for i in range(n)]),
        np.zeros(n),
        np.array(edges, dtype=np.int64).reshape(-1, 2),
        np.array(lengths, float),
    )


def expanded_optimum(graph, start, goals, risk, deadline):
    # Each transition strictly increases time, so this oracle is a finite DAG
    # DP independent of all label-setting and Dijkstra implementations.
    table = [[None] * graph.n_nodes for _ in range(deadline + 1)]
    table[0][start] = 0
    adjacency = [[] for _ in range(graph.n_nodes)]
    for (a, b), length in zip(graph.edges, graph.edge_cost, strict=True):
        steps = max(1, math.ceil(float(length) / 0.8 / 0.5))
        for source, target in ((a, b), (b, a)):
            cost = max(1, round((float(length) + risk[target] * steps * 0.5) * 65536))
            adjacency[source].append((target, steps, cost))
    for elapsed in range(deadline + 1):
        for node, cost in enumerate(table[elapsed]):
            if cost is None:
                continue
            for target, steps, weight in adjacency[node]:
                arrival = elapsed + steps
                if arrival > deadline:
                    continue
                candidate = cost + weight
                previous = table[arrival][target]
                if previous is None or candidate < previous:
                    table[arrival][target] = candidate
    candidates = [row[goal] for row in table for goal in goals if row[goal] is not None]
    return min(candidates) if candidates else None


def validate_path(graph, result, start, goals, risk, deadline):
    assert result.route[0] == start and result.route[-1] in goals
    lookup = {
        tuple(sorted(e)): float(d)
        for e, d in zip(graph.edges, graph.edge_cost, strict=True)
    }
    cost = elapsed = 0
    for a, b in pairwise(result.route):
        length = lookup[tuple(sorted((a, b)))]
        steps = max(1, math.ceil(length / 0.8 / 0.5))
        elapsed += steps
        cost += max(1, round((length + risk[b] * steps * 0.5) * 65536))
    assert elapsed == result.travel_steps <= deadline
    assert cost / 65536 == result.estimated_objective
    assert len(set(result.route)) == len(result.route)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_exact_methods_match_independent_time_expanded_dp(strategy):
    rng = np.random.default_rng(665927)
    for _ in range(200):
        n = int(rng.integers(4, 14))
        edges = [(i, i + 1) for i in range(n - 1)]
        edges += [
            (i, j) for i in range(n) for j in range(i + 2, n) if rng.random() < 0.2
        ]
        graph = graph_from_edges(n, edges, rng.uniform(0.1, 3.0, len(edges)))
        risk = rng.uniform(0, 10, n)
        goals = (n - 1, n - 2)
        deadline = int(rng.integers(1, 60))
        expected = expanded_optimum(graph, 0, goals, risk, deadline)
        result = deadline_route(
            graph,
            0,
            goals,
            risk,
            available_steps=deadline,
            walking_speed_m_s=0.8,
            dt_s=0.5,
            max_labels=5000,
            strategy=strategy,
        )
        assert result.feasible == (expected is not None)
        assert result.optimal
        if expected is not None:
            assert result.estimated_objective == expected / 65536
            assert result.certified_gap == 0.0
            validate_path(graph, result, 0, goals, risk, deadline)


@pytest.mark.parametrize("capacity", [1, 2, 8, 64])
def test_memory_capped_dual_search_keeps_feasibility_and_valid_gap(capacity):
    rng = np.random.default_rng(971)
    for _ in range(40):
        n = 9
        edges = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if j == i + 1 or rng.random() < 0.2
        ]
        graph = graph_from_edges(n, edges, rng.uniform(0.2, 2.0, len(edges)))
        risk = rng.uniform(0, 8, n)
        deadline = 20
        expected = expanded_optimum(graph, 0, (n - 1,), risk, deadline)
        arena = MCUArena(4096)
        prefix = arena.alloc(3)
        prefix[:] = b"abc"
        result = deadline_route(
            graph,
            0,
            (n - 1,),
            risk,
            available_steps=deadline,
            walking_speed_m_s=0.8,
            dt_s=0.5,
            max_labels=capacity,
            arena=arena,
            strategy="dual_astar",
        )
        assert arena.used == 3 and bytes(prefix) == b"abc" and arena.high_water <= 4096
        assert result.labels_used <= capacity
        if expected is not None:
            assert result.feasible
            validate_path(graph, result, 0, (n - 1,), risk, deadline)
            assert (
                result.lower_bound_q16
                <= expected
                <= round(result.estimated_objective * 65536)
            )
            if result.optimal:
                assert result.estimated_objective == expected / 65536
            else:
                assert result.resource_fallback


def test_tiny_arena_and_terminal_start_need_no_fake_optimality():
    graph = graph_from_edges(3, [(0, 1), (1, 2)], [1, 1])
    for strategy in STRATEGIES:
        terminal = deadline_route(
            graph,
            2,
            (2,),
            np.zeros(3),
            available_steps=0,
            walking_speed_m_s=1,
            dt_s=1,
            arena=MCUArena(1),
            strategy=strategy,
        )
        assert (
            terminal.optimal and terminal.certified_gap == 0 and terminal.route == (2,)
        )
        result = deadline_route(
            graph,
            0,
            (2,),
            np.zeros(3),
            available_steps=2,
            walking_speed_m_s=1,
            dt_s=1,
            arena=MCUArena(1),
            strategy=strategy,
        )
        assert result.feasible and result.resource_fallback and not result.optimal


def test_high_cost_path_certificate_uses_int64_without_label_overflow():
    graph = graph_from_edges(4, [(0, 1), (1, 2), (2, 3)], [1, 1, 1])
    risk = np.full(4, INT32_MAX / 65536 - 2)
    result = deadline_route(
        graph,
        0,
        (3,),
        risk,
        available_steps=3,
        walking_speed_m_s=1,
        dt_s=1,
        strategy="dual_astar",
    )
    assert result.optimal and result.feasible and result.certified_gap == 0
    assert result.estimated_objective * 65536 > INT32_MAX


@pytest.mark.parametrize("multiplier", [1000, 10000000])
def test_heuristic_tables_preserve_integer_ties_above_float_exact_range(multiplier):
    from ember.routing.constrained import _path, _table

    # The two complete paths differ by one Q16 cost unit; a float shortcut
    # outside its exact range could falsely choose the more costly path.
    reverse = [
        [],
        [(0, 1000000000, 1000000001)],
        [(0, 1000000000, 1000000000)],
        [(1, 1000000000, 1000000000), (2, 1000000000, 1000000000)],
    ]
    table = _table(reverse, (3,), MCUArena(4096), multiplier=multiplier)
    assert table[0][0] == 2000000000 + multiplier * 2000000000
    assert _path(table, 0, (3,)) == (0, 2, 3)
