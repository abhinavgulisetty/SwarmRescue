"""Connected abstraction, delivered hotspots and certified fine refinement."""

from dataclasses import replace
from itertools import pairwise

import mujoco
import numpy as np
import pytest

from cascade.extract import TraversabilityGraph, VoxelGrid
from ember.comms import packetize_gaussians
from ember.physical import PhysicalScan
from ember.radio import RadioConfig
from ember.regions import RegionMap, RegionRobotTeam
from mcu import MCUOutOfMemory
from tests.test_gbp import _line_graph
from tests.test_physical import _world


def _grid(width, height, rng=None):
    cells = np.array([(x, y, 0) for x in range(width) for y in range(height)])
    edges = []
    for x in range(width):
        for y in range(height):
            node = x * height + y
            for target in (
                node + height if x + 1 < width else -1,
                node + 1 if y + 1 < height else -1,
            ):
                if target >= 0 and (rng is None or rng.random() > 0.25):
                    edges.append((node, target))
    occupied = np.zeros((width, height, 1), dtype=bool)
    return TraversabilityGraph(
        VoxelGrid((0.0, 0.0, 0.0), (0.25, 0.25, 0.25), occupied, occupied),
        cells,
        np.full(len(cells), 0.25),
        np.array(edges).reshape(-1, 2),
        np.ones(len(edges)),
    )


def _components(graph):
    pending = set(range(graph.n_nodes))
    adjacent = graph.adjacency()
    groups = []
    while pending:
        nodes = {min(pending)}
        queue = list(nodes)
        while queue:
            for target, _ in adjacent[queue.pop()]:
                if target not in nodes:
                    nodes.add(target)
                    queue.append(target)
        pending -= nodes
        groups.append(frozenset(nodes))
    return set(groups)


@pytest.mark.parametrize("region_cells", (1, 2, 4, 8))
def test_partition_preserves_components_and_each_edge_has_a_fine_certificate(
    region_cells,
):
    fine = _grid(10, 9, np.random.default_rng(12))
    regions = RegionMap.build(fine, region_cells=region_cells)
    expected = {
        frozenset(int(regions.node_to_region[n]) for n in component)
        for component in _components(fine)
    }
    assert _components(regions.graph) == expected
    fine_edges = {tuple(sorted(edge)) for edge in fine.edges}
    positions = fine.world_positions()
    for (a, b), certificate, length in zip(
        regions.graph.edges, regions.certificates, regions.travel_lengths_m, strict=True
    ):
        assert certificate[0] == regions.anchors[a]
        assert certificate[-1] == regions.anchors[b]
        assert all(
            tuple(sorted((i, j))) in fine_edges for i, j in pairwise(certificate)
        )
        assert length == pytest.approx(
            sum(
                np.linalg.norm(positions[i] - positions[j])
                for i, j in pairwise(certificate)
            )
        )
    for members in regions.members:
        allowed = set(int(n) for n in members)
        visited = {int(members[0])}
        queue = list(visited)
        while queue:
            for target, _ in regions.adjacent[queue.pop()]:
                if target in allowed and target not in visited:
                    visited.add(target)
                    queue.append(target)
        assert visited == allowed


def test_disconnected_cells_in_one_spatial_block_are_not_merged():
    fine = _grid(4, 1)
    fine = replace(fine, edges=fine.edges[[0, 2]], edge_cost=np.ones(2))
    regions = RegionMap.build(fine, region_cells=4)
    assert regions.graph.n_nodes == 2
    assert regions.graph.n_edges == 0
    assert regions.node_to_region[0] != regions.node_to_region[3]


def test_peak_pooling_does_not_dilute_hotspots_or_count_samples_as_certainty():
    regions = RegionMap.build(_grid(4, 1), region_cells=4)
    scan = PhysicalScan(
        np.arange(4),
        np.array([0.1, 0.2, 10.0, 0.1]),
        np.array([0.1, 0.1, 0.8, 0.1]),
        np.arange(4) + 300.0,
        np.zeros(4),
    )
    peak = regions.aggregate(scan)
    np.testing.assert_array_equal(peak.nodes, [0])
    np.testing.assert_array_equal(peak.mean, [10.0])
    np.testing.assert_array_equal(peak.variance, [0.8])
    np.testing.assert_array_equal(peak.temperature_k, [302.0])


def test_cool_peer_or_duplicate_cannot_erase_a_delivered_hotspot():
    fine = _grid(4, 1)
    team = RegionRobotTeam(fine, fine.world_positions()[[0, 3]], loss_seed=8)
    robot = team.robots[0]
    hot = packetize_gaussians(1, 0, np.array([0]), np.array([10.0]), np.array([0.1]))[0]
    cool = packetize_gaussians(1, 1, np.array([0]), np.array([0.1]), np.array([0.1]))[0]
    for packet in (hot, cool, hot.forwarded()):
        robot.enqueue(packet)
    assert robot.consume() == 2
    assert robot.hazard_floor[0] == pytest.approx(10.0)
    assert robot.engine.belief().mean[0] < robot.hazard_floor[0]
    assert not np.shares_memory(robot.hazard_floor, team.robots[1].hazard_floor)


def test_changed_region_membership_discards_evidence_but_preserves_stable_region():
    fine = _line_graph(8)
    team = RegionRobotTeam(fine, fine.world_positions()[[0, 7]], loss_seed=8)
    robot = team.robots[0]
    robot.observe(np.array([0, 1]), np.array([10.0, 12.0]), np.array([0.1, 0.1]))
    robot.consume()
    changed = replace(
        fine, edges=np.delete(fine.edges, 1, axis=0), edge_cost=np.ones(6)
    )
    team.replace_graph(changed)
    assert team.discarded_region_factors == 1
    assert team.preserved_factors == 1
    assert team.geometry_revisions == 1
    assert team.robots[0].next_sequence == robot.next_sequence
    np.testing.assert_array_equal(team.robots[0].hazard_floor, [0.0, 0.0, 12.0])
    assert len(team.robots[0].seen_factors) == 1
    assert np.all(team.robots[0].engine.message_information == 0)


def test_grid_change_is_rejected_without_consuming_or_mutating_evidence():
    fine = _line_graph(8)
    team = RegionRobotTeam(fine, fine.world_positions()[[0, 7]], loss_seed=8)
    robot = team.robots[0]
    robot.observe(np.array([0]), np.array([10.0]), np.array([0.1]))
    before = robot.engine.local_information.copy()
    changed = replace(fine, grid=replace(fine.grid, origin=(1.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="voxel lattice"):
        team.replace_graph(changed)
    assert team.robots[0] is robot and len(robot.queue) == 1
    np.testing.assert_array_equal(robot.engine.local_information, before)


def test_numeric_budget_rejects_oversized_partition_without_expanding_arena():
    fine = _line_graph(4000)
    with pytest.raises(MCUOutOfMemory, match="regional"):
        RegionRobotTeam(
            fine, fine.world_positions()[[0, -1]], loss_seed=8, region_cells=1
        )
    bounded = RegionRobotTeam(
        fine, fine.world_positions()[[0, -1]], loss_seed=8, region_cells=4
    )
    assert all(robot.arena.capacity < 200 * 1024 for robot in bounded.robots)


def test_peer_hotspot_avoided_by_integer_regional_route_with_real_fine_edges():
    fine, domain = _world()
    positions = fine.world_positions()
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody>
        <geom name="g_wall_partition" type="box" pos="1 .5 .8" size=".05 .35 .8"/>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    state = domain.initial_state().ignite(
        domain,
        tuple(positions[1]),
        radius_m=0.51,
        temperature_k=900.0,
        smoke_kg_m3=0.25,
    )
    team = RegionRobotTeam(
        fine,
        positions[[0, 4]],
        loss_seed=8,
        region_cells=1,
        radio_config=RadioConfig(packet_error_rate=0),
    )
    team.observe_and_exchange(
        domain,
        state,
        positions[[0, 4]],
        positions[[4, 0]],
        np.random.default_rng(7),
        model=model,
        data=data,
        scan_radius_m=4.0,
        beta=1.0,
    )
    result = team.plan(
        0,
        domain,
        state,
        positions[0],
        positions[4],
        positions[[0, 1, 4]],
        np.random.default_rng(8),
        beta=1.0,
        available_time_s=20.0,
    )
    assert not team.robots[0].fine_observed[1]
    assert team.robots[1].fine_observed[1]
    assert result.converged and result.persistent_robot_belief
    np.testing.assert_array_equal(result.waypoints, positions[[0, 2, 3, 5, 4]])
    assert team.arena_high_water_bytes < 200 * 1024
    assert any(sender == 1 for sender, _, _ in team.robots[0].seen_factors)


def test_observer_exposure_matches_oracle_heat_smoke_without_base_cost():
    from cascade.demo import _robot_exposure_rate
    from ember.routing import OracleCostConfig, oracle_state_cost

    graph, domain = _world()
    state = domain.initial_state().ignite(
        domain, (1.5, 1.0, 0.5), radius_m=10.0, temperature_k=900.0, smoke_kg_m3=0.25
    )
    positions = graph.world_positions()
    expected = (
        oracle_state_cost(graph, domain, state) - OracleCostConfig().base_step_cost
    )
    np.testing.assert_allclose(_robot_exposure_rate(domain, state, positions), expected)
    np.testing.assert_array_equal(
        _robot_exposure_rate(domain, domain.initial_state(), positions),
        np.zeros(len(positions)),
    )


def test_exposure_handles_exterior_spawns_without_clamping_them_into_fire():
    from cascade.demo import _robot_exposure_rate

    _, domain = _world()
    state = domain.initial_state().ignite(
        domain, (1.5, 1.0, 0.5), radius_m=10.0, temperature_k=900.0, smoke_kg_m3=0.25
    )
    positions = np.array(
        [[-1.0, 0.5, 0.5], [1.0, 1.0, -0.1], [3.0, 1.0, 0.5], [1.0, 1.0, 0.5]]
    )
    rate = _robot_exposure_rate(domain, state, positions)
    np.testing.assert_array_equal(rate[:3], np.zeros(3))
    assert rate[3] > 0
    with pytest.raises(ValueError, match="finite"):
        _robot_exposure_rate(domain, state, np.array([[np.nan, 0.0, 0.0]]))


def test_fine_readings_choose_clear_path_inside_one_hot_region():
    fine, domain = _world()
    positions = fine.world_positions()
    state = domain.initial_state().ignite(
        domain,
        tuple(positions[1]),
        radius_m=0.51,
        temperature_k=900.0,
        smoke_kg_m3=0.25,
    )
    team = RegionRobotTeam(fine, positions[[0, 4]], loss_seed=8, region_cells=4)
    assert team.graph.n_nodes == 1
    scans = []
    for robot in team.robots:
        scan = team._scan(
            domain,
            state,
            positions[0],
            np.random.default_rng(7),
            robot=robot,
            scan_radius_m=4.0,
        )
        robot.observe(scan.nodes, scan.mean, scan.variance)
        robot.consume()
        robot.engine.run()
        scans.append(scan)
    team.scans = tuple(scans)
    result = team.plan(
        0,
        domain,
        state,
        positions[0],
        positions[4],
        positions[[0, 1, 4]],
        np.random.default_rng(8),
        beta=1.0,
        available_time_s=20.0,
    )
    np.testing.assert_array_equal(result.waypoints, positions[[0, 2, 3, 5, 4]])
    assert result.proxy_improved
    storage = np.frombuffer(team.robots[0].arena._storage, dtype=np.uint8)
    assert np.shares_memory(team.robots[0].fine_mean, storage)
    assert np.shares_memory(team.robots[0].fine_variance, storage)
    assert np.shares_memory(team.robots[0].fine_observed, storage)


def test_direct_fine_cache_remaps_only_surviving_sites_and_ages_variance():
    fine = _line_graph(8)
    team = RegionRobotTeam(fine, fine.world_positions()[[0, -1]], loss_seed=8)
    old = team.robots[0]
    old.fine_mean[:] = np.arange(8)
    old.fine_variance[:] = 0.1
    old.fine_observed[:] = True
    kept = np.array([0, 2, 3, 4, 5, 6, 7])
    mapping = np.full(8, -1)
    mapping[kept] = np.arange(7)
    edges = fine.edges[np.all(np.isin(fine.edges, kept), axis=1)]
    changed = replace(
        fine,
        cells=fine.cells[kept],
        surface_z=fine.surface_z[kept],
        edges=mapping[edges],
        edge_cost=np.ones(len(edges)),
    )
    team.replace_graph(changed, process_variance=0.02)
    new = team.robots[0]
    np.testing.assert_array_equal(new.fine_mean, kept)
    np.testing.assert_allclose(new.fine_variance, 0.12)
    assert np.all(new.fine_observed)
    assert not np.shares_memory(new.fine_mean, old.fine_mean)


def test_refinement_preserves_terrain_penalty_instead_of_taking_short_rubble_route():
    fine, _ = _world()
    fine = replace(fine, edge_cost=np.array([9.0, 9.0, 1.0, 1.0, 1.0, 1.0]))
    regions = RegionMap.build(fine, region_cells=4)
    nodes, _, count, expansions = regions.refine(0, 4, (0,), np.zeros(6))
    assert nodes == (0, 2, 3, 5, 4)
    assert count == 6 and expansions == 0


def test_clock_infeasible_corridor_expands_to_find_feasible_alternative():
    fine, _ = _world()
    regions = RegionMap.build(fine, region_cells=1)
    detour = tuple(int(regions.node_to_region[n]) for n in (0, 2, 3, 5, 4))
    nodes, clock, count, expansions = regions.refine(
        0, 4, detour, np.zeros(6), available_time_s=7.5
    )
    assert clock.feasible and clock.travel_steps <= 30
    assert nodes == (0, 1, 4)
    assert expansions == 1 and count == 6


def test_structural_map_change_detects_drift_rotation_and_removal():
    from ember.physical_team import structural_pose_change_m

    origin = {"beam": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))}
    radii = {"beam": 3.0}
    assert structural_pose_change_m(origin, origin, radii) == 0
    flipped = {"beam": ((0.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 0.0))}
    assert structural_pose_change_m(origin, flipped, radii) == 0
    drifted = {"beam": ((0.15, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))}
    assert structural_pose_change_m(origin, drifted, radii) == pytest.approx(0.15)
    # A long beam's tip moves substantially even with small center rotation.
    turned = {"beam": ((0.0, 0.0, 0.0), (np.cos(0.05), 0.0, 0.0, np.sin(0.05)))}
    assert structural_pose_change_m(origin, turned, radii) == pytest.approx(
        6 * np.sin(0.05)
    )
    assert np.isinf(structural_pose_change_m(origin, {}, radii))
