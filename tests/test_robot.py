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

from cascade.robot import RobotSpec, WaypointDriver, build_robot, place_robot


def _world(spec: RobotSpec) -> tuple[mujoco.MjModel, mujoco.MjData]:
    body, actuators = build_robot(spec, (0.0, 0.0, 0.2))
    root = ET.Element("mujoco")
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
    return np.array(
        [[0.25 * i, 0.0, 0.0] for i in range(length)], dtype=np.float64
    )


def test_a_driver_with_no_route_holds_station() -> None:
    spec = RobotSpec(name="probe")
    model, data = _world(spec)
    driver = WaypointDriver(spec=spec, waypoints=np.zeros((0, 3)))
    driver.step(model, data)
    assert driver.finished
    assert np.allclose(data.ctrl, 0.0)


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
        place_robot(model, data, spec, tuple(waypoint[:2]) + (0.2,))
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
