"""Tests for the presentation layer.

These look like cosmetic tests and are not.  Every defect they pin produced a
video in which the robots were simply not visible, which is indistinguishable
from the simulation not working -- and took far longer to diagnose than it
should have precisely because nothing was asserting on it.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from cascade.cinema import (
    PLAN_WALL_ALPHA,
    WALL_ALPHA,
    PathTrace,
    SwarmCamera,
    Trails,
    draw_paths,
    hero_style,
    index_geoms,
    plan_style,
    plan_view,
)

# A slab wide in x and y but thin in z, high up; a tall wall; a column; a wall
# on the storey above; and a robot.  The slab is the shape that broke the
# dollhouse cut; the upper wall is the shape that made the plan pane
# unreadable.
_SCENE = """
<mujoco>
  <worldbody>
    <geom name="g_slab_s1_b0_0" type="box" size="4 3 0.1" pos="4 3 3.0"/>
    <geom name="g_wall_s0_r0_x" type="box" size="0.075 2 1.2" pos="2 2 1.2"/>
    <geom name="g_wall_s1_r0_x" type="box" size="0.075 2 1.4" pos="3 2 4.5"/>
    <geom name="g_col_s0_n0" type="box" size="0.175 0.175 1.5" pos="1 1 1.5"/>
    <body name="robot0" pos="1 1 0.1">
      <freejoint name="robot0_root"/>
      <geom name="robot0_chassis" type="box" size="0.17 0.12 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_string(_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, index_geoms(model, data)


def _alpha(model: mujoco.MjModel, name: str) -> float:
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    return float(model.geom_rgba[geom][3])


def test_geoms_are_classified_by_member_kind(scene) -> None:
    model, _, groups = scene
    names = {
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)): n
        for n in (
            "g_slab_s1_b0_0",
            "g_wall_s0_r0_x",
            "g_wall_s1_r0_x",
            "g_col_s0_n0",
            "robot0_chassis",
        )
    }
    assert [names[int(g)] for g in groups.slabs] == ["g_slab_s1_b0_0"]
    assert sorted(names[int(g)] for g in groups.walls) == [
        "g_wall_s0_r0_x",
        "g_wall_s1_r0_x",
    ]
    assert [names[int(g)] for g in groups.frame] == ["g_col_s0_n0"]
    assert [names[int(g)] for g in groups.robots] == ["robot0_chassis"]


def test_low_z_is_the_vertical_extent_not_the_largest_one(scene) -> None:
    """The bug that rendered the dollhouse view with its roof on.

    The slab is 8 x 6 x 0.2 m centred at z = 3.  Its underside is at 2.9.
    Taking the largest half-extent instead of the vertical one puts it at
    -1.0, so it never counts as above any cut and is never removed.
    """
    model, _, groups = scene
    slab = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g_slab_s1_b0_0")
    assert groups.low_z[slab] == pytest.approx(2.9, abs=1e-6)


def test_hero_style_removes_everything_above_the_cut(scene) -> None:
    model, _, groups = scene
    hero_style(model, groups, ceiling_z=1.8)
    assert _alpha(model, "g_slab_s1_b0_0") == pytest.approx(0.0)
    assert _alpha(model, "g_wall_s0_r0_x") == pytest.approx(WALL_ALPHA)


def test_hero_style_keeps_a_high_cut_intact(scene) -> None:
    model, _, groups = scene
    hero_style(model, groups, ceiling_z=50.0)
    assert _alpha(model, "g_slab_s1_b0_0") > 0.5


def test_plan_style_removes_every_slab(scene) -> None:
    model, _, groups = scene
    plan_style(model, groups, floor_z=0.0, ceiling_z=50.0)
    assert _alpha(model, "g_slab_s1_b0_0") == pytest.approx(0.0)
    assert _alpha(model, "g_wall_s0_r0_x") == pytest.approx(PLAN_WALL_ALPHA)


def test_plan_style_shows_one_storey_only(scene) -> None:
    """A plan of four floors at once is not a plan of anything.

    Cutting from above alone left every storey below the subject in shot, so
    watching a robot on the top floor meant looking at four sets of partitions
    stacked in projection with the robot lost somewhere inside.
    """
    model, _, groups = scene
    plan_style(model, groups, floor_z=0.2, ceiling_z=1.95)
    assert _alpha(model, "g_wall_s0_r0_x") == pytest.approx(PLAN_WALL_ALPHA)
    assert _alpha(model, "g_wall_s1_r0_x") == pytest.approx(0.0)


def test_plan_style_hides_the_storey_below(scene) -> None:
    """The slice must not reach back down through this storey's floor."""
    model, _, groups = scene
    plan_style(model, groups, floor_z=3.2, ceiling_z=4.95)
    assert _alpha(model, "g_wall_s1_r0_x") == pytest.approx(PLAN_WALL_ALPHA)
    assert _alpha(model, "g_wall_s0_r0_x") == pytest.approx(0.0)


def test_plan_view_frames_the_storey_it_is_of(scene) -> None:
    """Aimed at the slice, so the plan is one size whichever floor it shows."""
    ground = plan_view((24.0, 18.0), floor_z=0.0)
    roof = plan_view((24.0, 18.0), floor_z=11.2)
    assert roof.lookat[2] - ground.lookat[2] == pytest.approx(11.2)
    assert roof.distance == pytest.approx(ground.distance)


def test_styling_never_hides_a_robot(scene) -> None:
    """Whatever else the camera does, the subject stays on screen."""
    model, _, groups = scene
    for apply in (
        lambda: hero_style(model, groups, 0.5),
        lambda: plan_style(model, groups, 0.0, 0.5),
    ):
        apply()
        assert _alpha(model, "robot0_chassis") > 0.9


# --- camera -------------------------------------------------------------------


def test_camera_aims_at_its_subject() -> None:
    camera = SwarmCamera(footprint_m=(24.0, 18.0))
    subject = np.array([4.0, 4.0, 0.1])
    others = np.array([[4.0, 4.0, 0.1], [6.0, 6.0, 0.1]])
    view = camera.update(subject, others)
    assert view.lookat[0] == pytest.approx(4.0, abs=1e-6)
    assert view.lookat[1] == pytest.approx(4.0, abs=1e-6)


def test_camera_ignores_robots_on_other_storeys() -> None:
    """A robot four floors up must not drag the framing off this one.

    It cannot be in shot: the dollhouse cut that exposes the subject's storey
    removes the one it is standing on.  Framing for it only pushes the camera
    back until the robots that *are* visible are too small to see.
    """
    camera = SwarmCamera(footprint_m=(24.0, 18.0))
    subject = np.array([4.0, 4.0, 0.1])
    others = np.array([[4.0, 4.0, 0.1], [20.0, 16.0, 12.0]])
    view = camera.update(subject, others)
    assert view.distance == pytest.approx(camera.min_distance_m, abs=1e-6)


def test_camera_never_pulls_back_past_its_limit() -> None:
    """A straggler must not drag the framing out until nothing is legible."""
    camera = SwarmCamera(footprint_m=(24.0, 18.0))
    subject = np.array([0.0, 0.0, 0.0])
    scattered = np.array([[0.0, 0.0, 0.0], [400.0, 400.0, 0.0]])
    view = camera.update(subject, scattered)
    assert view.distance <= camera.max_distance_m + 1e-9


def test_camera_orbits_between_frames() -> None:
    camera = SwarmCamera(footprint_m=(24.0, 18.0))
    subject = np.array([4.0, 4.0, 0.1])
    others = np.array([[4.0, 4.0, 0.1]])
    first = camera.update(subject, others).azimuth
    second = camera.update(subject, others).azimuth
    assert second != first


def test_ceiling_sits_above_the_subject() -> None:
    camera = SwarmCamera(footprint_m=(24.0, 18.0))
    assert camera.ceiling(np.array([6.0, 6.0, 6.2])) > 6.2


def test_ceiling_ignores_robots_above_the_subject() -> None:
    """Cutting above the *highest* robot seals the building over the rest.

    One robot reaching the roof put the cut at 13.75 m, above every slab in
    the model, so nothing was removed and the remaining three robots spent the
    rest of the run inside an opaque box.
    """
    camera = SwarmCamera(footprint_m=(24.0, 18.0))
    assert camera.ceiling(np.array([4.0, 4.0, 0.1])) < 3.0


# --- routes and trails --------------------------------------------------------


def test_trails_ignore_a_robot_that_is_standing_still() -> None:
    """A stalled robot must not spend the whole buffer on one spot.

    The buffer is shared with the robot that got furthest, and it is that
    robot's history that gets dropped when it fills.
    """
    trails = Trails(n_robots=1, min_step_m=0.1)
    for _ in range(50):
        trails.record(np.array([[1.0, 1.0, 0.0]]))
    assert trails.of(0).shape[0] == 1


def test_trails_follow_a_robot_that_moves() -> None:
    trails = Trails(n_robots=2, min_step_m=0.1)
    for step in range(10):
        trails.record(np.array([[step * 0.5, 0.0, 0.0], [0.0, step * 0.5, 0.0]]))
    assert trails.of(0).shape[0] == 10
    assert trails.of(1).shape[0] == 10


def test_trails_stay_inside_their_budget_and_keep_both_ends() -> None:
    """Overflow thins the history; it must not truncate the journey."""
    trails = Trails(n_robots=1, max_points=20, min_step_m=0.01)
    for step in range(500):
        trails.record(np.array([[step * 0.1, 0.0, 0.0]]))
    track = trails.of(0)
    assert track.shape[0] <= 20
    assert track[0][0] == pytest.approx(0.0)
    assert track[-1][0] == pytest.approx(49.9)


def test_draw_paths_adds_geometry_to_the_scene(scene) -> None:
    model, data, _ = scene
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        renderer.update_scene(data)
        before = renderer.scene.ngeom
        draw_paths(
            renderer.scene,
            [
                PathTrace(
                    livery=(1.0, 0.5, 0.1),
                    route=np.linspace([0, 0, 0], [4, 4, 0], 40),
                    trail=np.linspace([0, 0, 0], [2, 2, 0], 12),
                )
            ],
        )
        assert renderer.scene.ngeom > before
    finally:
        renderer.close()


def test_draw_paths_survives_a_full_geom_buffer(scene) -> None:
    """Running out of room must drop lines, not raise or corrupt the scene."""
    model, data, _ = scene
    renderer = mujoco.Renderer(model, height=64, width=64, max_geom=8)
    try:
        renderer.update_scene(data)
        draw_paths(
            renderer.scene,
            [
                PathTrace(
                    livery=(1.0, 0.5, 0.1),
                    route=np.linspace([0, 0, 0], [40, 40, 0], 4000),
                    trail=np.linspace([0, 0, 0], [20, 20, 0], 4000),
                )
            ],
        )
        assert renderer.scene.ngeom <= renderer.scene.maxgeom
    finally:
        renderer.close()
