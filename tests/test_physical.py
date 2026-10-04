"""The robot-side belief must be able to replace a physical route."""

from __future__ import annotations

import numpy as np

from cascade.extract import TraversabilityGraph, VoxelGrid
from ember.belief import HazardSensorConfig
from ember.physical import plan_belief_route
from hazard import HazardDomain


def _world() -> tuple[TraversabilityGraph, HazardDomain]:
    cells = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0))
    )
    occupied = np.zeros((3, 2, 1), dtype=bool)
    graph = TraversabilityGraph(
        grid=VoxelGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), occupied, occupied),
        cells=cells,
        surface_z=np.full(6, 0.5),
        edges=np.asarray(((0, 1), (1, 4), (0, 2), (2, 3), (3, 5), (5, 4))),
        edge_cost=np.ones(6),
    )
    active = np.ones((1, 2, 3), dtype=bool)
    domain = HazardDomain(
        origin_m=(0.0, 0.0, 0.0),
        bounds_max_m=(3.0, 2.0, 1.0),
        cell_size_m=1.0,
        active=active,
        connector=np.zeros_like(active),
        vertical_open=np.zeros_like(active),
        initial_fuel_kg_m3=np.zeros_like(active, dtype=np.float64),
    )
    return graph, domain


def test_noisy_hazard_belief_replaces_the_shortcut_with_a_safe_route() -> None:
    graph, domain = _world()
    positions = graph.world_positions()
    state = domain.initial_state().ignite(
        domain,
        tuple(positions[1]),
        radius_m=0.51,
        temperature_k=900.0,
        smoke_kg_m3=0.25,
    )

    result = plan_belief_route(
        graph,
        domain,
        state,
        positions[0],
        positions[4],
        positions[[0, 1, 4]],
        np.random.default_rng(7),
        scan_radius_m=4.0,
        sensor_config=HazardSensorConfig(
            temperature_noise_k=0.0,
            smoke_noise_kg_m3=0.0,
        ),
    )

    np.testing.assert_allclose(result.waypoints, positions[[0, 2, 3, 5, 4]])
    assert result.converged
    assert result.improved
    assert result.peak_smoke_kg_m3 > 0.0
    assert result.proxy_improved
    assert result.baseline_proxy_cost > result.planned_proxy_cost


def test_delivered_peer_evidence_drives_persistent_integer_physical_route() -> None:
    from ember.comms import packetize_gaussians
    from ember.robot_state import RobotState

    graph, domain = _world()
    positions = graph.world_positions()
    robot = RobotState(0, graph, positions[0])
    peer_packet = packetize_gaussians(
        1, 0, np.array([1]), np.array([6.0]), np.array([0.01]),
    )[0]
    robot.enqueue(peer_packet)
    result = plan_belief_route(
        graph, domain, domain.initial_state(), positions[0], positions[4],
        positions[[0, 1, 4]], np.random.default_rng(7), scan_radius_m=0.1,
        beta=1.0, robot_state=robot,
    )
    np.testing.assert_allclose(result.waypoints, positions[[0, 2, 3, 5, 4]])
    assert result.persistent_robot_belief
    assert result.routing_arithmetic == "integer_log_q16"
    assert result.converged
    assert not robot.queue
    assert (1, 0, 1) in robot.seen_factors
    previous_count = len(robot.seen_factors)
    plan_belief_route(
        graph, domain, domain.initial_state(), positions[0], positions[4],
        positions[[0, 1, 4]], np.random.default_rng(8), scan_radius_m=0.1,
        beta=1.0, robot_state=robot,
    )
    assert len(robot.seen_factors) == previous_count + 1
