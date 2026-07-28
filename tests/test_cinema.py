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
    RobotReadout,
    SwarmCamera,
    draw_hud,
    hero_style,
    index_geoms,
    plan_style,
    stack_panes,
)

# A slab wide in x and y but thin in z, high up; a tall wall; a column; and a
# robot.  The slab is the shape that broke the dollhouse cut.
_SCENE = """
<mujoco>
  <worldbody>
    <geom name="g_slab_s1_b0_0" type="box" size="4 3 0.1" pos="4 3 3.0"/>
    <geom name="g_wall_s0_r0_x" type="box" size="0.075 2 1.2" pos="2 2 1.2"/>
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
            "g_col_s0_n0",
            "robot0_chassis",
        )
    }
    assert [names[int(g)] for g in groups.slabs] == ["g_slab_s1_b0_0"]
    assert [names[int(g)] for g in groups.walls] == ["g_wall_s0_r0_x"]
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
    plan_style(model, groups, ceiling_z=50.0)
    assert _alpha(model, "g_slab_s1_b0_0") == pytest.approx(0.0)
    assert _alpha(model, "g_wall_s0_r0_x") == pytest.approx(PLAN_WALL_ALPHA)


def test_styling_never_hides_a_robot(scene) -> None:
    """Whatever else the camera does, the subject stays on screen."""
    model, _, groups = scene
    for apply in (
        lambda: hero_style(model, groups, 0.5),
        lambda: plan_style(model, groups, 0.5),
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


# --- overlay ------------------------------------------------------------------


def test_hud_adds_bands_without_covering_the_frame() -> None:
    """The caption must not sit on top of the thing it is captioning.

    Drawn as an overlay, the title bar and the status panel between them hid
    the top and bottom third of the plan pane -- the view that exists to show
    where every robot is.
    """
    frame = np.full((240, 480, 3), 200, dtype=np.uint8)
    out = draw_hud(
        frame,
        title="t",
        subtitle="s",
        readouts=[
            RobotReadout("robot0", (1.0, 0.5, 0.1), 0, 3, 0.4, "driving"),
        ],
        elapsed_s=1.0,
    )
    assert out.dtype == np.uint8
    assert out.shape[1] == frame.shape[1]
    assert out.shape[0] > frame.shape[0]
    # Every row of the original survives somewhere, untouched.
    rows = {tuple(row) for row in out.reshape(out.shape[0], -1)}
    assert tuple(np.full(480 * 3, 200, dtype=np.uint8)) in rows
    kept = sum(1 for row in out.reshape(out.shape[0], -1) if row.min() == 200)
    assert kept == frame.shape[0]


def test_hud_actually_draws_something() -> None:
    frame = np.zeros((240, 480, 3), dtype=np.uint8)
    out = draw_hud(
        frame,
        title="EMBER",
        subtitle="s",
        readouts=[RobotReadout("robot0", (1.0, 0.5, 0.1), 0, 3, 0.4, "driving")],
        elapsed_s=1.0,
    )
    assert out.any(), "overlay left the frame untouched"


def test_stack_panes_joins_side_by_side() -> None:
    left = np.zeros((100, 160, 3), dtype=np.uint8)
    right = np.zeros((50, 50, 3), dtype=np.uint8)
    out = stack_panes(left, right)
    assert out.shape[0] == 100
    assert out.shape[1] > left.shape[1]
