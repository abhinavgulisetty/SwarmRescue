"""Independent receiver beliefs, provenance, buffers and deadlines."""

from __future__ import annotations

import numpy as np
import pytest

from cascade.extract import TraversabilityGraph, VoxelGrid
from ember.comms import packetize_gaussians
from ember.robot_state import RobotState


def _graph() -> TraversabilityGraph:
    occupied = np.zeros((3, 1, 1), dtype=bool)
    return TraversabilityGraph(
        VoxelGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), occupied, occupied),
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
        np.zeros(3),
        np.array([[0, 1], [1, 2]]),
        np.ones(2),
    )


def _packet(sequence=7):
    return packetize_gaussians(
        0,
        sequence,
        np.array([0]),
        np.array([4.0]),
        np.array([0.1]),
    )[0]


def test_partitioned_receivers_keep_independent_beliefs() -> None:
    graph = _graph()
    first = RobotState(0, graph, np.zeros(3))
    second = RobotState(1, graph, np.ones(3))
    initial_second = second.engine.belief().mean.copy()
    first.enqueue(_packet())
    for _ in range(100):
        first.tick()
    assert first.engine.belief().mean[0] > 1.0
    np.testing.assert_array_equal(second.engine.belief().mean, initial_second)
    assert not np.shares_memory(
        first.engine.local_information, second.engine.local_information
    )
    # Reconnection brings the missing observation to the second receiver.
    second.enqueue(_packet())
    for _ in range(100):
        second.tick()
    np.testing.assert_allclose(
        second.engine.belief().mean, first.engine.belief().mean, atol=1e-8
    )


def test_forwarded_factor_is_counted_once_and_new_sequence_is_distinct() -> None:
    robot = RobotState(0, _graph(), np.zeros(3))
    packet = _packet()
    robot.enqueue(packet)
    robot.enqueue(packet.forwarded())
    assert robot.consume() == 1
    before = robot.engine.local_precision.copy()
    robot.enqueue(packet)
    assert robot.consume() == 0
    np.testing.assert_array_equal(before, robot.engine.local_precision)
    robot.enqueue(_packet(sequence=8))
    assert robot.consume() == 1
    assert robot.engine.local_precision[0] > before[0]


def test_deadline_defers_queue_and_sweep_without_mutation() -> None:
    robot = RobotState(0, _graph(), np.zeros(3), cycles_per_tick=1)
    initial = robot.engine.local_information.copy()
    robot.enqueue(_packet())
    assert robot.tick() is False
    assert robot.deadline_misses == 1
    assert len(robot.queue) == 1
    np.testing.assert_array_equal(initial, robot.engine.local_information)
    robot.cycles_per_tick = 10000
    retained_buffer = robot.engine.message_precision
    assert robot.tick() is True
    assert robot.engine.message_precision is retained_buffer
    assert robot.inference_sweeps == 1
    assert not robot.queue


def test_factor_and_queue_capacities_fail_closed() -> None:
    robot = RobotState(0, _graph(), np.zeros(3), queue_capacity=1, factor_capacity=1)
    robot.enqueue(_packet())
    with pytest.raises(BufferError, match="queue"):
        robot.enqueue(_packet(sequence=8))
    robot.consume()
    before = robot.engine.local_precision.copy()
    robot.enqueue(_packet(sequence=8))
    with pytest.raises(BufferError, match="provenance"):
        robot.consume()
    np.testing.assert_array_equal(before, robot.engine.local_precision)


def test_replanning_reuses_the_robot_arena_without_growing_storage() -> None:
    from ember.routing import LMDPConfig, build_lmdp

    graph = _graph()
    robot = RobotState(0, graph, np.zeros(3))
    model = build_lmdp(graph, np.full(3, 0.3), (2,), config=LMDPConfig())
    first = robot.plan(model)
    used = robot.arena.used
    storage = np.frombuffer(robot.arena._storage, dtype=np.uint8)
    assert np.shares_memory(first.negative_log_desirability, storage)
    for _ in range(4):
        assert robot.plan(model).route(0) == (0, 1, 2)
        assert robot.arena.used == used
    assert robot.arena.high_water < robot.arena.capacity


@pytest.mark.parametrize("nodes", (1, 3, 19, 4000))
def test_compact_storage_estimate_matches_actual_arena_and_routes(nodes):
    from ember.gbp import gbp_storage_bytes
    from ember.physical_team import PhysicalRobotTeam
    from ember.routing import LMDPConfig, build_lmdp
    from tests.test_gbp import _line_graph

    graph = _line_graph(nodes)
    team = PhysicalRobotTeam(graph, np.zeros((2, 3)), loss_seed=11)
    robot = team.robots[0]
    assert robot.arena.used == gbp_storage_bytes(graph.n_nodes, graph.n_edges)
    numeric = sum(
        getattr(robot.engine, name).nbytes
        for name in (
            "local_precision",
            "local_information",
            "source",
            "target",
            "reverse",
            "edge_variance",
            "message_precision",
            "message_information",
        )
    )
    legacy = 16 * nodes + 96 * graph.n_edges
    assert numeric == 16 * nodes + 60 * graph.n_edges
    assert numeric <= legacy
    # Nonzero self-loop weights exercise the transition-count workspace bound.
    model = build_lmdp(
        graph,
        np.full(nodes, 0.3),
        (nodes - 1,),
        config=LMDPConfig(self_loop_weight=0.5, max_sweeps=20_000),
    )
    assert robot.plan(model, native=True).route(0) == tuple(range(nodes))
    assert robot.arena.used <= robot.arena.capacity
    assert not np.shares_memory(robot.engine.source, team.robots[1].engine.source)


def test_explicit_factor_backpressure_preserves_history_and_drains_queue():
    robot = RobotState(
        0, _graph(), np.zeros(3), factor_capacity=1, drop_on_factor_overflow=True
    )
    robot.enqueue(_packet())
    assert robot.consume() == 1
    before = robot.engine.local_precision.copy()
    robot.enqueue(_packet(sequence=8))
    assert robot.consume() == 0
    assert robot.dropped_factors == 1
    assert len(robot.seen_factors) == 1 and not robot.queue
    np.testing.assert_array_equal(before, robot.engine.local_precision)
    robot.enqueue(_packet())
    assert robot.consume() == 0
    assert robot.dropped_factors == 1
    np.testing.assert_array_equal(before, robot.engine.local_precision)
