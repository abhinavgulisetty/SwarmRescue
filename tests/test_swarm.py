"""Tests for swarm assignment and the waypoint driver.

Built on synthetic graphs and a bare robot rather than a generated building,
so they stay fast enough to run on every change.  The building-scale claims
belong in `test_extract.py`, which owns reachability.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest

from cascade.extract import TraversabilityGraph, VoxelGrid
from cascade.floorplan import build_floorplan
from cascade.params import BuildingParams
from cascade.robot import RobotSpec, RobotState, WaypointDriver, build_robot
from cascade.swarm import (
    GOAL_FLOOR_SEARCH_BUDGET,
    SETTLED_HOLD_STEPS,
    SETTLED_SPEED_M_S,
    Assignment,
    Swarm,
    _entry_approach,
    _spawn_pose,
    choose_goals,
)

STOREY_HEIGHT_M = 3.0


def _graph(storeys: int = 4, per_storey: int = 12) -> TraversabilityGraph:
    """A ladder of storeys, each a straight run of cells along x."""
    cell = (0.25, 0.25, 0.1)
    origin = (0.0, 0.0, -0.1)
    empty = np.zeros((per_storey, 1, 1), dtype=bool)
    grid = VoxelGrid(origin=origin, cell=cell, occupied=empty, blocked=empty)

    cells, surface = [], []
    for storey in range(storeys):
        for i in range(per_storey):
            cells.append((i, 0, 0))
            surface.append(storey * STOREY_HEIGHT_M)
    return TraversabilityGraph(
        grid=grid,
        cells=np.asarray(cells, dtype=np.int32),
        surface_z=np.asarray(surface, dtype=np.float64),
        edges=np.zeros((0, 2), dtype=np.int32),
        edge_cost=np.zeros(0, dtype=np.float64),
    )


# --- goal assignment ----------------------------------------------------------


def test_offset_queue_slots_line_up_square_with_the_entrance() -> None:
    plan = build_floorplan(7, BuildingParams())
    spec = RobotSpec()
    for slot in range(4):
        spawn, _ = _spawn_pose(plan, 0, slot, spec.spawn_height_m)
        approach = _entry_approach(plan, 0, spawn, spec.ride_height_m)
        door = plan.exits[0]

        np.testing.assert_allclose(approach[0, :2], spawn[:2])
        if door.axis.value == "x":
            assert approach[-1, 1] == pytest.approx(door.center[1])
            assert approach[-2, 1] == pytest.approx(door.center[1])
        else:
            assert approach[-1, 0] == pytest.approx(door.center[0])
            assert approach[-2, 0] == pytest.approx(door.center[0])


def test_goals_are_spread_across_storeys() -> None:
    """The property that makes the demo a building search, not a floor search.

    Sampling the reachable set without stratifying puts most goals on the
    ground storey purely because most reachable cells are there.
    """
    graph = _graph()
    cost = np.arange(graph.n_nodes, dtype=np.float64)
    goals = choose_goals(graph, cost, 4, STOREY_HEIGHT_M)

    storeys = {
        int(np.floor(graph.surface_z[g] / STOREY_HEIGHT_M + 0.25)) for g in goals
    }
    assert len(goals) == 4
    assert len(storeys) == 4, f"goals bunched onto storeys {sorted(storeys)}"


def test_goals_prefer_upper_storeys_first() -> None:
    """With fewer robots than storeys, the swarm should still go up."""
    graph = _graph()
    cost = np.arange(graph.n_nodes, dtype=np.float64)
    goals = choose_goals(graph, cost, 1, STOREY_HEIGHT_M)
    assert graph.surface_z[goals[0]] == pytest.approx(3.0 * STOREY_HEIGHT_M)


def test_goal_search_cost_is_bounded_beyond_the_nearest_landing() -> None:
    graph = _graph(storeys=1, per_storey=4)
    cost = np.asarray((10.0, 20.0, 45.0, 100.0))

    goal = choose_goals(graph, cost, 1, STOREY_HEIGHT_M)[0]

    assert cost[goal] <= cost.min() + GOAL_FLOOR_SEARCH_BUDGET
    assert goal == 2, "the bound collapsed the mission to the nearest cell"


def test_unreachable_nodes_are_never_chosen() -> None:
    graph = _graph()
    cost = np.full(graph.n_nodes, np.inf)
    reachable = 5
    cost[reachable] = 1.0
    assert choose_goals(graph, cost, 3, STOREY_HEIGHT_M) == [reachable]


def test_no_goals_when_nothing_is_reachable() -> None:
    graph = _graph()
    cost = np.full(graph.n_nodes, np.inf)
    assert choose_goals(graph, cost, 4, STOREY_HEIGHT_M) == []


# --- the driver ---------------------------------------------------------------


def _robot_world(spec: RobotSpec) -> tuple[mujoco.MjModel, mujoco.MjData]:
    body, actuators = build_robot(spec, (0.0, 0.0, spec.spawn_height_m))
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep="0.002")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", name="ground", type="plane", size="40 40 0.1")
    world.append(body)
    block = ET.SubElement(root, "actuator")
    for element in actuators:
        block.append(element)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _straight_route(n: int = 20) -> np.ndarray:
    return np.stack([np.linspace(1.0, 6.0, n), np.zeros(n), np.zeros(n)], axis=1)


def _assignment(name: str = "robot") -> Assignment:
    spec = RobotSpec(name=name)
    return Assignment(
        spec=spec,
        spawn=(0.0, 0.0, spec.spawn_height_m),
        yaw_rad=0.0,
        route=_straight_route(),
        goal_storey=0,
    )


def test_at_rest_requires_placement_support_uprightness_and_low_speed() -> None:
    assignment = _assignment("probe")
    model, data = _robot_world(assignment.spec)
    swarm = Swarm([assignment])

    with pytest.raises(RuntimeError, match="place the swarm"):
        swarm.at_rest(data)
    swarm.place(model, data)
    assert not swarm.at_rest(data), "motionless but airborne is not settled"

    steps, settled = swarm.settle(model, data, max_steps=2_000)
    assert settled
    assert SETTLED_HOLD_STEPS < steps < 2_000
    assert swarm.at_rest(data)

    member = swarm.members[0]
    data.qvel[member.dof_adr] = 2.0 * SETTLED_SPEED_M_S
    assert not swarm.at_rest(data)

    data.qvel[member.dof_adr : member.dof_adr + 6] = 0.0
    joint = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"{assignment.name}_root",
    )
    qpos_adr = model.jnt_qposadr[joint]
    angle = 0.5 * np.pi
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = (
        np.cos(0.5 * angle),
        np.sin(0.5 * angle),
        0.0,
        0.0,
    )
    mujoco.mj_forward(model, data)
    assert not swarm.at_rest(data)


def test_settle_reports_timeout_when_an_airborne_robot_cannot_land() -> None:
    assignment = _assignment("probe")
    model, data = _robot_world(assignment.spec)
    model.opt.gravity[:] = 0.0
    swarm = Swarm([assignment])
    swarm.place(model, data)

    max_steps = SETTLED_HOLD_STEPS + 5
    assert swarm.settle(model, data, max_steps=max_steps) == (max_steps, False)


def test_focus_prefers_deployed_robots_and_keeps_fallen_robots_active() -> None:
    swarm = Swarm([_assignment("first"), _assignment("queued")])
    deployed, queued = swarm.members
    deployed.driver.index = 8
    deployed.driver._state = RobotState.FALLEN
    queued.driver.index = 0
    queued.driver._state = RobotState.WAITING

    assert swarm.focus() is deployed
    assert not swarm.done, "a recoverable fall ended the swarm lifecycle"

    deployed.driver.stalled = True
    assert swarm.focus() is queued
    queued.driver.stalled = True
    assert swarm.done


def test_focus_breaks_equal_progress_ties_by_assignment_order() -> None:
    swarm = Swarm([_assignment("first"), _assignment("second")])
    for member in swarm.members:
        member.driver._state = RobotState.DRIVING
        member.driver.index = 4
    assert swarm.focus() is swarm.members[0]


def test_swarm_can_use_a_short_departure_stagger_for_a_bounded_demo() -> None:
    swarm = Swarm(
        [_assignment("first"), _assignment("second"), _assignment("third")],
        departure_stagger_steps=150,
    )

    delays = [member.driver.start_delay_steps for member in swarm.members]
    assert delays == [0, 150, 300]


def test_swarm_rejects_a_negative_departure_stagger() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Swarm([_assignment("probe")], departure_stagger_steps=-1)


def test_driver_holds_still_through_its_start_delay() -> None:
    """Staggered departure is what keeps a swarm out of its own doorway."""
    spec = RobotSpec()
    model, data = _robot_world(spec)
    driver = WaypointDriver(
        spec=spec, waypoints=_straight_route(), start_delay_steps=50
    )

    for _ in range(50):
        driver.step(model, data)
        assert np.allclose(data.ctrl, 0.0), "moved before its delay elapsed"

    driver.step(model, data)
    assert np.any(data.ctrl != 0.0), "never set off after the delay"


def test_driver_with_no_route_is_failed_and_idle() -> None:
    spec = RobotSpec()
    model, data = _robot_world(spec)
    driver = WaypointDriver(spec=spec, waypoints=np.zeros((0, 3)))
    assert not driver.finished
    assert driver.stalled
    assert driver.state is RobotState.UNROUTABLE

    driver.step(model, data)
    assert not driver.finished
    assert driver.stalled
    assert driver.state is RobotState.UNROUTABLE
    assert np.allclose(data.ctrl, 0.0)


def test_driver_slows_for_a_corner() -> None:
    """Cornering speed, not route quality, is what wedged the first swarm.

    A 0.34 m chassis cannot hold a right-angle turn through a doorway at
    cruise; it overshoots into the wall past the corner and rides up it.
    """
    spec = RobotSpec()
    along_x = np.stack([np.linspace(0.0, 2.0, 9), np.zeros(9), np.zeros(9)], axis=1)
    corner = along_x.copy()
    corner[5:, 0] = corner[4, 0]
    corner[5:, 1] = np.linspace(0.25, 1.0, 4)

    straight = WaypointDriver(spec=spec, waypoints=along_x, index=2)
    turning = WaypointDriver(spec=spec, waypoints=corner, index=2)

    assert straight._corner_factor(4) == pytest.approx(1.0, abs=1e-6)
    assert turning._corner_factor(4) < 0.75


def test_corner_factor_never_stops_the_robot_dead() -> None:
    spec = RobotSpec()
    reversal = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    driver = WaypointDriver(spec=spec, waypoints=reversal, index=0)
    assert driver._corner_factor(2) >= driver.min_speed_factor


def test_driver_reports_arrival_at_the_end_of_its_route() -> None:
    spec = RobotSpec()
    model, data = _robot_world(spec)
    route = _straight_route(4)
    driver = WaypointDriver(spec=spec, waypoints=route)

    # Put the robot on the final waypoint and let it notice.
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{spec.name}_root")
    adr = model.jnt_qposadr[joint]
    data.qpos[adr : adr + 3] = route[-1] + np.array([0.0, 0.0, spec.ride_height_m])
    mujoco.mj_forward(model, data)

    driver.step(model, data)
    assert driver.finished
    assert driver.state is RobotState.ARRIVED


def test_robots_do_not_collide_with_each_other() -> None:
    """A Phase 0 simplification, asserted so it cannot regress silently.

    Robots must still collide with the world -- the moment they stop doing
    that, the traversal demo proves nothing at all.
    """
    spec = RobotSpec()
    model, _ = _robot_world(spec)
    chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{spec.name}_chassis")
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

    def collides(a: int, b: int) -> bool:
        return bool(
            model.geom_contype[a] & model.geom_conaffinity[b]
            or model.geom_contype[b] & model.geom_conaffinity[a]
        )

    assert not collides(chassis, chassis), "robots would pile into each other"
    assert collides(chassis, ground), "robot would fall through the world"
