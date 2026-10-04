"""Live peer evidence, bounded transport and geometry-epoch correctness."""

from dataclasses import replace
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from cascade.extract import GRID_MARGIN_M, TraversabilityGraph, rasterize
from cascade.geometry import Rect
from cascade.mjcf import settled_poses
from cascade.params import BuildingParams, CascadeParams
from cascade.robot import RobotSpec, build_robot
from cascade.structure import Member, MemberKind, Structure
from cascade.swarm import Assignment, Swarm
from ember.belief import HazardSensorConfig
from ember.comms import packetize_gaussians
from ember.physical import plan_belief_route
from ember.physical_team import (
    PhysicalRobotTeam,
    collision_aware_params,
    ground_mission_graph,
    place_collision_free_team,
    rebuild_physical_graph,
    select_source_packets,
)
from ember.radio import RadioConfig
from ember.routing import build_lmdp
from ember.voi import solve_adjoint, solve_adjoint_sparse
from tests.test_physical import _world


def test_visible_peer_scan_routes_robot_around_its_occluded_hazard():
    graph, domain = _world()
    positions = graph.world_positions()
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
        temperature_k=900,
        smoke_kg_m3=0.25,
    )
    team = PhysicalRobotTeam(
        graph,
        positions[[0, 4]],
        loss_seed=13,
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
        scan_radius_m=4,
        beta=1,
    )
    assert 1 not in team.scans[0].nodes
    assert 1 in team.scans[1].nodes
    assert any(
        sender == 1 and cell == 1 for sender, _, cell in team.robots[0].seen_factors
    )
    assert team.radio_stats.delivered > 0 and team.max_wall_count == 1
    result = team.plan(
        0,
        domain,
        state,
        positions[0],
        positions[4],
        positions[[0, 1, 4]],
        np.random.default_rng(8),
        beta=1,
    )
    assert result.persistent_robot_belief and result.converged
    np.testing.assert_allclose(result.waypoints, positions[[0, 2, 3, 5, 4]])


def test_selection_keeps_factor_identity_and_charges_each_source_header():
    graph, _ = _world()
    team = PhysicalRobotTeam(graph, graph.world_positions()[[0, 4]], loss_seed=2)
    cells = np.arange(40)
    packets = packetize_gaussians(0, 17, cells, np.ones(40), np.ones(40))
    scores = np.zeros(40)
    scores[[0, 19, 38]] = [3, 2, 1]
    selected = select_source_packets(packets, cells, scores, 40)
    assert sum(p.size_bytes for p in selected) == 40
    assert [(p.sequence, p.cells.tolist()) for p in selected] == [(17, [0]), (18, [19])]
    robot = team.robots[0]
    own = robot.observe(np.array([1]), np.array([3.0]), np.array([0.1]))
    robot.consume()
    before = robot.engine.local_precision.copy()
    robot.enqueue(own[0].forwarded())
    assert robot.consume() == 0
    np.testing.assert_array_equal(robot.engine.local_precision, before)


def test_rebuild_preserves_direct_factors_and_epochs_without_reusing_messages():
    graph, _ = _world()
    positions = graph.world_positions()
    team = PhysicalRobotTeam(graph, positions[[0, 4]], loss_seed=5)
    robot = team.robots[0]
    robot.observe(np.array([1, 2]), np.array([6.0, 2.0]), np.array([0.1, 0.2]))
    robot.tick()
    before_precision = robot.engine.local_precision.copy()
    before_information = robot.engine.local_information.copy()
    order = np.array([5, 4, 3, 2, 0])  # remove node 1 and permute surviving IDs
    inverse = {int(old): new for new, old in enumerate(order)}
    edges = [
        (inverse[int(a)], inverse[int(b)])
        for a, b in graph.edges
        if int(a) in inverse and int(b) in inverse
    ]
    rebuilt = TraversabilityGraph(
        graph.grid,
        graph.cells[order],
        graph.surface_z[order],
        np.asarray(edges),
        np.ones(len(edges)),
    )
    mapping = team.replace_graph(rebuilt, process_variance=0)
    new = team.robots[0]
    np.testing.assert_allclose(new.engine.local_precision, before_precision[order])
    np.testing.assert_allclose(new.engine.local_information, before_information[order])
    assert np.all(new.engine.message_precision == 0)
    assert new.seen_factors == {(0, 0, mapping[2])}
    assert new.next_sequence == robot.next_sequence == 1
    new.observe(np.array([mapping[2]]), np.array([2.0]), np.array([0.2]))
    new.consume()
    assert (0, 1, mapping[2]) in new.seen_factors
    assert team.geometry_revisions == 1 and not team.scans


def test_live_graph_snapshot_uses_exact_moved_pose_and_excludes_robots():
    building = replace(BuildingParams(), footprint_m=(4.0, 4.0))
    params = replace(CascadeParams(), building=building)
    member = Member(
        "wall_1", MemberKind.WALL, (2.0, 2.0, 1.0), (0.2, 1.0, 2.0), 0, 1000.0
    )
    structure = Structure(building, (member,), ())
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody>
        <geom type="plane" size="10 10 .1"/>
        <body name="wall_1" pos="2 2 1"><joint type="slide" axis="1 0 0"/>
          <geom name="g_wall_1" type="box" size=".1 .5 1"/></body>
        <body name="robot0" pos=".5 .5 .2"><joint type="slide" axis="0 1 0"/>
          <geom name="robot0_chassis" type="box" size=".3 .3 .2"/></body>
      </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    data.qpos[0] = 0.08
    mujoco.mj_forward(model, data)
    authored = settled_poses(model, data, structure)
    exact = settled_poses(model, data, structure, snap_to_authored=False)
    assert authored["wall_1"][0][0] == 2.0
    assert exact["wall_1"][0][0] == 2.08
    first = rebuild_physical_graph(model, data, structure, params)
    data.qpos[1] = 2.0
    mujoco.mj_forward(model, data)
    robot_moved = rebuild_physical_graph(model, data, structure, params)
    np.testing.assert_array_equal(first.cells, robot_moved.cells)
    data.qpos[0] = 1.0
    mujoco.mj_forward(model, data)
    wall_moved = rebuild_physical_graph(model, data, structure, params)
    assert not np.array_equal(first.grid.blocked, wall_moved.grid.blocked)


def test_inference_blocks_respect_tick_budget_and_still_converge():
    graph, domain = _world()
    positions = graph.world_positions()
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    team = PhysicalRobotTeam(graph, positions[[0, 4]], loss_seed=91)
    for robot in team.robots:
        # Whole sweep needs 1,032 modeled cycles; only two message updates fit.
        robot.cycles_per_tick = 400
    team.observe_and_exchange(
        domain,
        domain.initial_state(),
        positions[[0, 4]],
        positions[[4, 0]],
        np.random.default_rng(71),
        model=model,
        data=data,
        scan_radius_m=0.1,
    )
    assert all(robot.deadline_misses == 0 for robot in team.robots)
    assert all(robot.last_cycles <= 400 for robot in team.robots)
    assert all(robot.inference_sweeps > 6 for robot in team.robots)


def test_direct_adjoint_matches_iterative_terminal_flow_and_priority():
    graph, _ = _world()
    model = build_lmdp(graph, np.full(graph.n_nodes, 0.3), (4,))
    source = np.zeros(graph.n_nodes)
    source[0] = 1
    iterative = solve_adjoint(model, source, tolerance=1e-12, max_sweeps=4000)
    direct = solve_adjoint_sparse(model, source)
    assert direct.converged and iterative.converged
    np.testing.assert_allclose(direct.field, iterative.field, atol=1e-10)
    assert direct.field[4] > 0 and direct.sweeps == 0


def test_penetrating_spawn_moves_to_clear_reachable_exterior_cell():
    original, _ = _world()
    positions = original.world_positions()
    positions[:, 2] = 0
    retained = np.array([1, 2, 3, 4, 5])
    graph = TraversabilityGraph(
        original.grid,
        original.cells[retained],
        np.zeros(5),
        np.array([[0, 3], [1, 2], [2, 4], [4, 3]]),
        np.ones(4),
    )
    spec = RobotSpec(name="robot0")
    spawn = (0.5, 0.5, spec.spawn_height_m)
    body, actuators = build_robot(spec, spawn)
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", type="plane", size="10 10 .1")
    ET.SubElement(
        world, "geom", name="fallen_beam", type="box", pos=".5 .5 .2", size=".3 .3 .2"
    )
    world.append(body)
    block = ET.SubElement(root, "actuator")
    for actuator in actuators:
        block.append(actuator)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    swarm = Swarm([Assignment(spec, spawn, 0, positions[[0, 1, 4]], 0)])
    swarm.place(model, data)
    assert place_collision_free_team(swarm, model, data, graph, Rect(1, 0, 3, 2)) == 1
    member = swarm.members[0]
    np.testing.assert_allclose(member.assignment.route[0], positions[2])
    np.testing.assert_allclose(member.assignment.route[-1], positions[4])
    assert all(
        c.dist >= -0.001
        or ((c.geom1 in member.geom_ids) == (c.geom2 in member.geom_ids))
        for c in data.contact[: data.ncon]
    )


def test_exterior_map_covers_eight_robot_queue_without_changing_default():
    params = CascadeParams()
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="10 10 .1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    default = rasterize(model, data, (4, 4), 2, params.traversability)
    expanded = rasterize(model, data, (4, 4), 2, params.traversability, margin_m=4)
    assert default.origin[:2] == (-GRID_MARGIN_M, -GRID_MARGIN_M)
    assert expanded.origin[:2] == (-4, -4)
    assert default.shape[2] == expanded.shape[2]
    assert expanded.origin[0] < -3.4 < default.origin[0]


def test_ground_scope_remaps_edges_without_inventing_shortcuts():
    graph, _ = _world()
    heights = graph.surface_z.copy()
    heights[2] = 3.0
    elevated = replace(graph, surface_z=heights)
    ground = ground_mission_graph(elevated)
    assert ground.n_nodes == 5
    assert np.all(ground.surface_z <= 1)
    old_edges = {
        tuple(sorted((tuple(graph.cells[a]), tuple(graph.cells[b]))))
        for a, b in graph.edges
    }
    for a, b in ground.edges:
        assert (
            tuple(sorted((tuple(ground.cells[a]), tuple(ground.cells[b])))) in old_edges
        )
    assert not any(np.array_equal(cell, graph.cells[2]) for cell in ground.cells)


def test_physical_deadline_uses_metres_instead_of_flat_cell_equivalents():
    graph, domain = _world()
    graph = replace(graph, edge_cost=np.full(graph.n_edges, 4.0))
    positions = graph.world_positions()
    state = domain.initial_state().ignite(
        domain,
        tuple(positions[1]),
        radius_m=0.51,
        temperature_k=900,
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
        scan_radius_m=4,
        beta=1,
        sensor_config=HazardSensorConfig(temperature_noise_k=0, smoke_noise_kg_m3=0),
        available_time_s=2.1,
        planning_speed_m_s=1,
        planning_dt_s=0.1,
    )
    assert result.deadline_intervened and result.modeled_deadline_feasible
    assert result.estimated_travel_time_s == 2
    np.testing.assert_allclose(result.waypoints, positions[[0, 1, 4]])


def test_planning_envelope_includes_wheels_and_mast_but_not_visual_beacon():
    params = CascadeParams()
    corrected = collision_aware_params(params, RobotSpec())
    assert corrected.traversability.robot_radius_m > 0.30
    assert 0.39 < corrected.traversability.min_clearance_m < 0.5
    assert (
        corrected.traversability.min_clearance_m > params.traversability.min_clearance_m
    )
    assert collision_aware_params(corrected, RobotSpec()) == corrected
