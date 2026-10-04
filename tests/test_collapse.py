"""Tests for the collapse stage, and mostly for its instrumentation.

The collapse is the one part of Cascade that claims to be *evidence*: the
rubble is not authored, so the energy trace is what says the solver produced it
rather than invented it.  That makes a wrong energy number worse than no energy
number, and the defect these tests pin produced one.
"""

from __future__ import annotations

import mujoco
import pytest

from cascade.collapse import (
    CACHE_VERSION,
    CollapseResult,
    apply_settled_state,
    cache_path,
    kinetic_energy,
    model_hash,
    pulverize,
)
from cascade.damage import BlastDamage

#: Two free bodies well clear of each other and of the ground, so nothing in
#: these tests depends on contact.
_SCENE = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom name="ground" type="plane" size="0 0 1"/>
    <body name="col_doomed" pos="0 0 4">
      <freejoint/>
      <geom type="box" size="0.175 0.175 1.5" density="2400"/>
    </body>
    <body name="col_spared" pos="6 0 4">
      <freejoint/>
      <geom type="box" size="0.175 0.175 1.5" density="2400"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def scene():
    model = mujoco.MjModel.from_xml_string(_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _body(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def test_a_pulverized_member_stays_where_it_was_put(scene) -> None:
    """Destroyed members are parked below the world and must *stay* there.

    They keep their degrees of freedom -- a MuJoCo model's body count is fixed
    at compile time -- so parking them without cancelling their weight leaves
    them in free fall for the entire collapse, accelerating away below the
    ground plane for as long as the simulation runs.
    """
    model, data = scene
    pulverize(model, data, ("col_doomed",))
    mujoco.mj_forward(model, data)
    body = _body(model, "col_doomed")
    parked = float(data.xpos[body][2])

    for _ in range(4000):  # the full eight seconds of a real collapse
        mujoco.mj_step(model, data)

    assert float(data.xpos[body][2]) == pytest.approx(parked, abs=1e-6)


def test_pulverized_mass_contributes_no_kinetic_energy(scene) -> None:
    """The defect this pins made the collapse's headline number meaningless.

    Seed 7 reported a peak kinetic energy of 177.3048 MJ.  The 90 columns the
    blast destroyed weigh 57.6 t, and 57.6 t in free fall for eight seconds
    carries 177.3047 MJ.  The agreement is to seven figures: the number being
    presented as evidence that the solver produced the rubble field was, in
    fact, entirely the sound of deleted mass falling forever.
    """
    model, data = scene
    pulverize(model, data, ("col_doomed",))
    mujoco.mj_forward(model, data)

    peak = 0.0
    for _ in range(4000):
        mujoco.mj_step(model, data)
        peak = max(peak, kinetic_energy(model, data))

    # The spared column falls 2.5 m onto the ground and lands; that is the only
    # energy in this scene.  Free fall of the destroyed one would be ~1.6 MJ.
    doomed_mass = float(model.body_mass[_body(model, "col_doomed")])
    free_fall = 0.5 * doomed_mass * (9.81 * 8.0) ** 2
    assert peak < 0.05 * free_fall


def test_a_spared_member_still_falls(scene) -> None:
    """The weight cancellation must not leak onto members that survived."""
    model, data = scene
    pulverize(model, data, ("col_doomed",))
    mujoco.mj_forward(model, data)
    body = _body(model, "col_spared")
    started = float(data.xpos[body][2])

    for _ in range(500):
        mujoco.mj_step(model, data)

    assert float(data.xpos[body][2]) < started - 0.5


def test_pulverize_ignores_members_that_are_not_there(scene) -> None:
    model, data = scene
    pulverize(model, data, ("col_does_not_exist",))  # must not raise


def test_model_hash_ignores_presentation(scene) -> None:
    """Recolouring a wall must not discard a cached collapse."""
    plain = "<mujoco><worldbody><geom type='box' size='1 1 1'/></worldbody></mujoco>"
    styled = (
        "<mujoco><asset><material name='m' rgba='1 0 0 1'/></asset>"
        "<worldbody><geom type='box' size='1 1 1' rgba='0 1 0 1'/></worldbody></mujoco>"
    )
    assert model_hash(plain) == model_hash(styled)


def test_cache_path_carries_the_format_version() -> None:
    """A stale keyframe from an older energy model must not be reused."""
    assert f"_v{CACHE_VERSION}_" in cache_path(7, "abc123").name


def test_settled_structure_can_prefix_a_model_with_appended_joints(scene) -> None:
    model, data = scene
    extra_qpos = data.qpos[7:].copy()
    settled = CollapseResult(
        seed=7,
        damage=BlastDamage((0.0, 0.0, 0.0), 0.0, (), (), ()),
        qpos=data.qpos[:7].copy(),
        steps_simulated=0,
        settled_early=True,
        wall_time_s=0.0,
        peak_kinetic_energy_j=0.0,
        final_kinetic_energy_j=0.0,
        model_hash="test",
    )
    settled.qpos[2] += 1.0

    with pytest.raises(ValueError, match="cache is stale"):
        apply_settled_state(model, data, settled)
    apply_settled_state(model, data, settled, allow_appended_joints=True)

    assert data.qpos[2] == settled.qpos[2]
    assert data.qpos[7:] == pytest.approx(extra_qpos)
