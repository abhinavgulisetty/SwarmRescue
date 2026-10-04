"""Geometry sensing must track live poses rather than visual presentation."""

import mujoco
import numpy as np

from ember.geometry_sensing import box_raycasts, radio_wall_counts


def _scene():
    model = mujoco.MjModel.from_xml_string('''
    <mujoco><worldbody>
      <body name="wall" pos="2 0 1">
        <joint type="slide" axis="0 1 0"/>
        <geom name="g_wall_1" type="box" size="0.1 1 1" rgba="1 1 1 0"/>
      </body>
      <geom name="robot_chassis" type="box" pos="1 0 1" size="0.2 0.2 0.2"/>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_invisible_wall_blocks_optical_ray_and_attenuates_radio() -> None:
    model, data = _scene()
    points = np.array([[0., 0., 1.], [4., 0., 1.], [0., 3., 1.]])
    hit, counts = box_raycasts(model, data, points[0], points[1:])
    assert abs(hit[0] - 1.9) < 1e-10
    assert counts.tolist() == [1, 0]
    assert np.isinf(hit[1])
    matrix = radio_wall_counts(model, data, points)
    assert matrix[0, 1] == matrix[1, 0] == 1
    assert np.all(np.diag(matrix) == 0)


def test_sensor_rays_track_displaced_and_disabled_geometry() -> None:
    model, data = _scene()
    points = np.array([[4., 0., 1.]])
    data.qpos[0] = 3.0
    mujoco.mj_forward(model, data)
    hit, walls = box_raycasts(model, data, [0., 0., 1.], points)
    assert np.isinf(hit[0]) and walls[0] == 0
    data.qpos[0] = 0.0
    mujoco.mj_forward(model, data)
    wall = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_wall_1")
    model.geom_contype[wall] = model.geom_conaffinity[wall] = 0
    hit, walls = box_raycasts(model, data, [0., 0., 1.], points)
    assert np.isinf(hit[0]) and walls[0] == 0
