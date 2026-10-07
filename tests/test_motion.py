"""Physical ground tracking must be continuous, bounded and collision-aware."""

from itertools import pairwise
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest

from cascade.motion import SmoothWaypointDriver, clear_segment, simplify_route
from cascade.robot import RobotSpec, build_robot


def world(*, box=False):
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep=str(1 / 600), integrator="implicitfast")
    body = ET.SubElement(root, "worldbody")
    ET.SubElement(body, "geom", type="plane", size="10 10 .1")
    if box:
        ET.SubElement(
            body, "geom", name="g_wall_test", type="box", pos="1 1 .5", size=".2 .2 .5"
        )
    spec = RobotSpec(name="robot0")
    robot, actuators = build_robot(spec, (0, 0, spec.spawn_height_m))
    body.append(robot)
    ET.SubElement(root, "actuator").extend(actuators)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, 360)
    return model, data, spec


def test_shortcut_cannot_cut_through_a_wall_or_its_robot_clearance():
    model, data, _ = world(box=True)
    route = np.array([[0, 1, 0], [0, 2, 0], [2, 2, 0], [2, 1, 0]], float)
    assert not clear_segment(model, data, route[0], route[-1])
    assert not clear_segment(model, data, [0, 1.4, 0], [2, 1.4, 0])
    simplified = simplify_route(model, data, route)
    for start, end in pairwise(simplified):
        assert clear_segment(model, data, start, end, radius_m=0.38)
    np.testing.assert_array_equal(simplified[[0, -1]], route[[0, -1]])


def test_ground_turn_tracks_without_circling_and_limits_wheel_changes():
    model, data, spec = world()
    route = np.array([[0, 0, 0], [1.5, 0, 0], [1.5, 1.5, 0]], float)
    driver = SmoothWaypointDriver(spec, route, arrive_radius_m=0.15)
    previous = data.xpos[model.body("robot0").id].copy()
    travel = 0
    for _ in range(12000):
        before = data.ctrl.copy()
        driver.step(model, data)
        assert (
            np.max(np.abs(data.ctrl - before))
            <= (driver.braking_rad_s2 + driver.turning_acceleration_rad_s2)
            * model.opt.timestep
            + 1e-10
        )
        mujoco.mj_step(model, data)
        point = data.xpos[model.body("robot0").id].copy()
        travel += np.linalg.norm(point[:2] - previous[:2])
        previous = point
        if driver.finished:
            break
    assert driver.finished
    assert np.linalg.norm(point[:2] - route[-1, :2]) < 0.16
    assert travel < 3.3
    assert abs(point[2] - spec.ride_height_m) < 0.01


def test_replanning_preserves_wheel_command_continuity_and_emergency_stop():
    model, data, spec = world()
    driver = SmoothWaypointDriver(spec, np.array([[0, 0, 0], [3, 0, 0]], float))
    for _ in range(300):
        driver.step(model, data)
        mujoco.mj_step(model, data)
    before = data.ctrl.copy()
    assert driver.reroute(np.array([data.xpos[model.body("robot0").id], [3, 1, 0]]))
    driver.step(model, data)
    assert np.max(np.abs(data.ctrl - before)) < 0.03
    driver.pause(model, data, emergency=True)
    np.testing.assert_array_equal(data.ctrl, np.zeros(4))


@pytest.mark.parametrize(
    "field", ["lookahead_m", "acceleration_rad_s2", "braking_rad_s2"]
)
def test_invalid_motion_limits_are_rejected(field):
    with pytest.raises(ValueError, match="finite and positive"):
        SmoothWaypointDriver(RobotSpec(), np.zeros((1, 3)), **{field: 0})
