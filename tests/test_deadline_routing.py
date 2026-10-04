"""Deadline feasibility, Pareto optimality and bounded-memory fallback."""

from itertools import pairwise, permutations

import numpy as np
import pytest

from cascade.extract import TraversabilityGraph, VoxelGrid
from ember.routing.deadline import deadline_route
from ember.voi import TokenBucket, schedule_coverage_updates, solve_policy_occupancy
from mcu import MCUArena
from mcu.log_lmdp import solve_log_q16


def _graph():
    occupied = np.zeros((5, 1, 1), dtype=bool)
    return TraversabilityGraph(
        VoxelGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), occupied, occupied),
        np.array([[i, 0, 0] for i in range(5)]),
        np.zeros(5),
        np.array([[0, 1], [1, 4], [0, 2], [2, 3], [3, 4]]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    )


def test_deadline_preserves_fast_route_when_safe_detour_is_too_slow():
    graph = _graph()
    risk = np.array([0.0, 10.0, 0.0, 0.0, 0.0])
    relaxed = deadline_route(
        graph,
        0,
        (4,),
        risk,
        available_steps=3,
        walking_speed_m_s=1.0,
        dt_s=1.0,
    )
    urgent = deadline_route(
        graph,
        0,
        (4,),
        risk,
        available_steps=2,
        walking_speed_m_s=1.0,
        dt_s=1.0,
    )
    assert relaxed.route == (0, 2, 3, 4)
    assert urgent.route == (0, 1, 4)
    assert relaxed.feasible and urgent.feasible
    impossible = deadline_route(
        graph,
        0,
        (4,),
        risk,
        available_steps=1,
        walking_speed_m_s=1.0,
        dt_s=1.0,
    )
    assert not impossible.feasible and impossible.route == urgent.route
    capped = deadline_route(
        graph,
        0,
        (4,),
        risk,
        available_steps=3,
        walking_speed_m_s=1.0,
        dt_s=1.0,
        max_labels=1,
    )
    assert capped.feasible and capped.resource_fallback and not capped.optimal
    assert capped.route == urgent.route


def test_pareto_router_matches_exhaustive_simple_path_optimum():
    graph = _graph()
    edges = {tuple(sorted(edge)) for edge in graph.edges.tolist()}
    rng = np.random.default_rng(101)
    for _ in range(20):
        risk = rng.uniform(0, 4, 5)
        horizon = int(rng.integers(2, 5))
        paths = []
        for count in range(4):
            for interior in permutations((1, 2, 3), count):
                path = (0, *interior, 4)
                if len(path) - 1 <= horizon and all(
                    tuple(sorted(edge)) in edges for edge in pairwise(path)
                ):
                    # Match the declared Q16 objective, including edge rounding.
                    cost = sum(round((1 + risk[target]) * 65536) for target in path[1:])
                    paths.append(cost / 65536)
        result = deadline_route(
            graph,
            0,
            (4,),
            risk,
            available_steps=horizon,
            walking_speed_m_s=1.0,
            dt_s=1.0,
        )
        assert result.optimal and result.feasible
        assert result.estimated_objective == min(paths)


def test_conditioned_occupancy_keeps_downstream_route_relevance():
    from ember.routing import build_lmdp

    graph = _graph()
    model = build_lmdp(graph, np.full(5, 3.0), (4,))
    solution = solve_log_q16(model)
    source = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    field = solve_policy_occupancy(model, solution.negative_log_desirability, source)
    assert abs(field[4] - 1.0) < 1e-8
    assert field[0] >= 1.0
    assert np.all(field >= 0.0)


def test_coverage_avoids_spending_entire_packet_on_one_redundant_cluster():
    positions = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [10.0, 0.0, 0.0]])
    result = schedule_coverage_updates(
        sender=0,
        sequence=0,
        cells=np.arange(3),
        prior_mean=np.zeros(3),
        prior_variance=np.ones(3),
        posterior_mean=np.full(3, 2.0),
        posterior_variance=np.full(3, 0.1),
        relevance=np.ones(3),
        positions_m=positions,
        bucket=TokenBucket(capacity_bytes=32, refill_bytes=0),
    )
    assert len(result.selected_cells) == 2
    assert 2 in result.selected_cells
    assert result.bytes_scheduled <= 32


def test_safeguard_preserves_feasible_routes_and_replaces_only_timeouts():
    from ember.routing.deadline import safeguard_route_deadline

    graph = _graph()
    risk = np.array([0.0, 10.0, 0.0, 0.0, 0.0])
    proposal = (0, 2, 3, 4)
    unchanged = safeguard_route_deadline(
        graph,
        0,
        (4,),
        risk,
        proposal,
        available_steps=3,
        walking_speed_m_s=1.0,
        dt_s=1.0,
    )
    assert unchanged.route == proposal and not unchanged.proposal_replaced
    rescued = safeguard_route_deadline(
        graph,
        0,
        (4,),
        risk,
        proposal,
        available_steps=2,
        walking_speed_m_s=1.0,
        dt_s=1.0,
    )
    assert rescued.feasible and rescued.route == (0, 1, 4)
    assert rescued.proposal_replaced


@pytest.mark.parametrize("remaining", (0, 1, 16, 17, 34, 1024, 4096 * 17))
def test_deadline_scratch_respects_remaining_arena_and_restores_prefix(remaining):
    graph = _graph()
    arena = MCUArena(4 + remaining)
    prefix = arena.alloc(3)
    prefix[:] = b"abc"
    for _ in range(3):
        result = deadline_route(
            graph,
            0,
            (4,),
            np.array([0.0, 10.0, 0.0, 0.0, 0.0]),
            available_steps=3,
            walking_speed_m_s=1.0,
            dt_s=1.0,
            arena=arena,
        )
        assert result.feasible
        assert result.travel_steps <= 3
        assert result.labels_used <= remaining // 17
        assert arena.used == 3
        assert arena.high_water <= arena.capacity
        assert bytes(prefix) == b"abc"
        if remaining < 34:
            assert result.resource_fallback and not result.optimal
            assert result.route == (0, 1, 4)
        if remaining >= 1024:
            assert result.optimal and not result.resource_fallback
            assert result.route == (0, 2, 3, 4)
