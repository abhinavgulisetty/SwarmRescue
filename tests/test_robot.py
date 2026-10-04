"""Behaviour of the waypoint driver.

These are not about control quality -- pure pursuit on a skid-steer chassis is
not the interesting part of this project.  They pin the bookkeeping around it,
where every defect so far has been: a robot that arrived and then drove away
again, and a robot that recovered from a snag and was still scored as stalled.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest

from cascade.robot import (
    SPAWN_CLEARANCE_M,
    RobotSpec,
    RobotState,
    WaypointDriver,
    build_robot,
    place_robot,
)


def _world(spec: RobotSpec) -> tuple[mujoco.MjModel, mujoco.MjData]:
    body, actuators = build_robot(spec, (0.0, 0.0, spec.spawn_height_m))
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", type="plane", size="20 20 0.1")
    world.append(body)
    block = ET.SubElement(root, "actuator")
    for actuator in actuators:
        block.append(actuator)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _straight_route(spec: RobotSpec, length: int = 12) -> np.ndarray:
    return np.array([[0.25 * i, 0.0, 0.0] for i in range(length)], dtype=np.float64)


@pytest.mark.parametrize(
    ("wheel_radius_m", "body_height_m"),
    [(0.13, 0.10), (0.08, 0.20)],
)
def test_spawn_height_is_derived_from_the_lowest_wheel(
    wheel_radius_m: float, body_height_m: float
) -> None:
    spec = RobotSpec(
        name="probe",
        wheel_radius_m=wheel_radius_m,
        body_height_m=body_height_m,
    )
    body, _ = build_robot(spec, (0.0, 0.0, spec.spawn_height_m))
    wheel = body.find("body[@name='probe_wheel_lf']")
    assert wheel is not None
    geom = wheel.find("geom")
    assert geom is not None

    body_z = float(body.attrib["pos"].split()[2])
    wheel_z = float(wheel.attrib["pos"].split()[2])
    radius = float(geom.attrib["size"].split()[0])
    assert body_z + wheel_z - radius == pytest.approx(SPAWN_CLEARANCE_M)
    assert spec.spawn_height_m > spec.ride_height_m


def test_a_driver_with_no_route_holds_station() -> None:
    spec = RobotSpec(name="probe")
    model, data = _world(spec)
    driver = WaypointDriver(spec=spec, waypoints=np.zeros((0, 3)))
    assert not driver.finished
    assert driver.stalled
    assert driver.state is RobotState.UNROUTABLE

    driver.step(model, data)
    assert not driver.finished, "an unreachable goal was reported as an arrival"
    assert driver.stalled
    assert driver.state is RobotState.UNROUTABLE
    assert np.allclose(data.ctrl, 0.0)
    with pytest.raises(ValueError, match="no target"):
        driver.target()


def test_waiting_driver_transitions_to_driving_after_its_delay() -> None:
    spec = RobotSpec(name="probe")
    model, data = _world(spec)
    driver = WaypointDriver(
        spec=spec,
        waypoints=_straight_route(spec),
        start_delay_steps=2,
    )

    assert driver.state is RobotState.WAITING
    for _ in range(2):
        driver.step(model, data)
        assert driver.state is RobotState.WAITING
        assert np.allclose(data.ctrl, 0.0)

    driver.step(model, data)
    assert driver.state is RobotState.DRIVING
    assert np.any(data.ctrl != 0.0)


def test_driver_reports_climbing_from_the_route_ahead() -> None:
    spec = RobotSpec(name="probe")
    model, data = _world(spec)
    route = np.array([[0.25 * i, 0.0, 0.10 * i] for i in range(8)], dtype=np.float64)
    driver = WaypointDriver(spec=spec, waypoints=route)

    driver.step(model, data)
    assert driver.state is RobotState.CLIMBING


def test_recovery_state_starts_with_a_recovery_command() -> None:
    spec = RobotSpec(name="probe")
    model, data = _world(spec)
    driver = WaypointDriver(
        spec=spec,
        waypoints=_straight_route(spec),
        recovery_after=1,
        recovery_steps=2,
    )

    driver.step(model, data)
    driving_command = data.ctrl.copy()
    driver.step(model, data)

    assert driver.state is RobotState.RECOVERING
    assert not np.allclose(data.ctrl, driving_command)


def test_fallen_state_is_recoverable_and_does_not_end_the_driver() -> None:
    spec = RobotSpec(name="probe")
    model, data = _world(spec)
    route = _straight_route(spec)
    route[:, 2] = 1.0
    driver = WaypointDriver(spec=spec, waypoints=route)

    driver.step(model, data)
    assert driver.state is RobotState.FALLEN
    assert not driver.finished
    assert not driver.stalled

    first = route[0]
    place_robot(
        model,
        data,
        spec,
        (float(first[0]), float(first[1]), float(first[2] + spec.ride_height_m)),
    )
    driver.step(model, data)
    assert driver.state is RobotState.DRIVING
    assert not driver.finished
    assert not driver.stalled


def test_fall_is_latched_before_reacquiring_a_lower_route() -> None:
    """A nearby lower flight must not rewrite a fall into ordinary driving."""
    spec = RobotSpec(name="probe")
    model, data = _world(spec)
    route = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    driver = WaypointDriver(
        spec=spec,
        waypoints=route,
        index=2,
        search_back=0,
        reacquire_m=0.5,
    )

    driver.step(model, data)

    assert driver.index == 0, "test did not exercise global re-acquisition"
    assert driver.state is RobotState.FALLEN
    assert not driver.stalled


def test_storey_fall_rewinds_past_the_bounded_index_floor() -> None:
    """A robot on a lower slab must restart at the foot of its lost flight."""
    spec = RobotSpec(name="probe")
    model, data = _world(spec)
    route = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.8],
            [3.0, 0.0, 1.6],
            [4.0, 0.0, 2.4],
            [5.0, 0.0, 3.0],
            [6.0, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    driver = WaypointDriver(
        spec=spec,
        waypoints=route,
        index=5,
        search_back=0,
        reacquire_back=2,
        reacquire_m=0.5,
    )
    driver._best_index = 6

    driver.step(model, data)

    assert driver.index == 0
    assert driver._best_index == 0
    assert driver.state is RobotState.FALLEN


def test_no_progress_transitions_to_stalled() -> None:
    spec = RobotSpec(name="probe")
    model, data = _world(spec)
    route = _straight_route(spec) + np.array([50.0, 0.0, 0.0])
    driver = WaypointDriver(spec=spec, waypoints=route, stall_patience=2)

    for _ in range(3):
        driver.step(model, data)

    assert driver.stalled
    assert not driver.finished
    assert driver.state is RobotState.STALLED


def test_aim_uses_full_lookahead_on_a_straight_route() -> None:
    spec = RobotSpec(name="probe")
    driver = WaypointDriver(
        spec=spec,
        waypoints=_straight_route(spec),
        lookahead=3,
    )
    assert driver._aim_index(np.zeros(3)) == 3


def test_aim_preserves_a_corner_after_projection_advances_to_it() -> None:
    """The indexed corner remains mandatory until the chassis reaches it."""
    spec = RobotSpec(name="probe")
    route = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    driver = WaypointDriver(spec=spec, waypoints=route, index=1, lookahead=2)

    assert driver._aim_index(np.array([0.5, 0.0, 0.0])) == 1


def test_aim_releases_a_projected_corner_inside_its_arrival_radius() -> None:
    """Ordinary lattice points must not become stop-and-go targets."""
    spec = RobotSpec(name="probe")
    route = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    driver = WaypointDriver(spec=spec, waypoints=route, index=1, lookahead=2)

    assert driver._aim_index(np.array([0.7, 0.0, 0.0])) == 2


def test_arriving_latches() -> None:
    """The failure this pins put a robot on the roof and then took it away.

    The projection runs before the arrival test, so a driver parked on its
    goal re-projected onto the last few waypoints, decided it was not there
    yet, and set off again -- one robot reached storey 4, turned round, and
    drove back down two storeys before the run ended.
    """
    spec = RobotSpec(name="probe")
    model, data = _world(spec)
    driver = WaypointDriver(spec=spec, waypoints=_straight_route(spec))
    driver.index = len(driver.waypoints)
    assert driver.finished

    for _ in range(50):
        driver.step(model, data)
        mujoco.mj_step(model, data)
    assert driver.finished, "arrival came undone"
    assert driver.state is RobotState.ARRIVED
    assert np.allclose(data.ctrl, 0.0), "a finished driver is still commanding"


def test_progress_defers_the_stall_indefinitely() -> None:
    """A robot that keeps advancing is never declared stalled, however slowly.

    The old logic zeroed the no-progress counter whenever a recovery began,
    which made the "no progress for `stall_patience` steps" test unreachable
    and left a compound condition that latched `stalled` on the first recovery
    attempt after the patience expired.  Robots that backed out of a snag,
    recovered and drove another twenty metres were still scored as stalled,
    and the run stopped as soon as every robot had been tarred once.
    """
    spec = RobotSpec(name="probe")
    model, data = _world(spec)
    route = _straight_route(spec, length=40)
    driver = WaypointDriver(spec=spec, waypoints=route, stall_patience=50)

    for waypoint in route[:30]:
        place_robot(model, data, spec, (*tuple(waypoint[:2]), 0.2))
        for _ in range(40):
            driver.step(model, data)
        assert not driver.stalled, "gave up on a robot that was still moving"


def test_a_wedged_driver_eventually_gives_up() -> None:
    spec = RobotSpec(name="probe")
    model, data = _world(spec)
    # A route that starts far away and that the robot is never driven toward.
    route = np.array([[50.0 + 0.25 * i, 0.0, 0.0] for i in range(12)])
    driver = WaypointDriver(spec=spec, waypoints=route, stall_patience=50)
    for _ in range(200):
        driver.step(model, data)
    assert driver.stalled


def test_reroute_reopens_a_driver_and_resets_projection() -> None:
    spec = RobotSpec(name="probe")
    driver = WaypointDriver(spec=spec, waypoints=_straight_route(spec))
    driver.index = len(driver.waypoints)
    driver.stalled = True
    replacement = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    driver.reroute(replacement)

    np.testing.assert_array_equal(driver.waypoints, replacement)
    assert driver.index == 0
    assert not driver.stalled
    assert driver.state is RobotState.WAITING


def test_identical_remaining_route_with_on_segment_pose_preserves_control_state():
    spec = RobotSpec(name="probe")
    route = np.array(
        [
            [11.125, 6.875, 0.0],
            [11.125, 7.125, 0.0],
            [11.375, 7.125, 0.0],
            [12.375, 7.125, 0.0],
        ]
    )
    driver = WaypointDriver(spec=spec, waypoints=route, index=1)
    driver._best_index = 1
    driver._since_progress = 27
    driver._recovery_left = 4
    original_buffer = driver.waypoints
    current_pose = np.array([11.1741063, 7.0288741, 0.01278395])
    assert driver.reroute(np.vstack((current_pose, route[1:]))) is False
    assert driver.waypoints is original_buffer and driver.index == 1
    assert driver._best_index == 1 and driver._since_progress == 27
    assert driver._recovery_left == 4
    assert driver.reroute(route[1:]) is False


def test_off_segment_prefix_or_changed_path_restarts_controller():
    spec = RobotSpec(name="probe")
    route = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    driver = WaypointDriver(spec=spec, waypoints=route, index=1)
    driver._since_progress = 27
    replacement = np.vstack(([5.0, 5.0, 0.0], route[1:]))
    assert driver.reroute(replacement) is True
    assert driver.index == 0 and driver._since_progress == 0
    np.testing.assert_array_equal(driver.waypoints, replacement)
    replacement[-1] = [1.0, 2.0, 0.0]
    assert driver.reroute(replacement) is True
