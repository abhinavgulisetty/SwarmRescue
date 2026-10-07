"""Rescue requires physical motion, contact ownership and a clear exit."""

from dataclasses import replace
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
import pytest

from ember.evacuation import (
    EvacuationConfig,
    EvacuationTeam,
    NoEvacuationRouteError,
    exterior_route,
    survivor_bodies,
)


def _scene(*, wall=False, count=1):
    root = ET.Element("mujoco")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world, "geom", type="plane", size="10 10 0.1", friction="0.08 0.01 0.001"
    )
    if wall:
        ET.SubElement(
            world, "geom", name="g_wall_test", type="box", size="0.1 3 1", pos="0.9 1 1"
        )
    for body in survivor_bodies(np.array([[1.8, 1 + i, 0] for i in range(count)])):
        world.append(body)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, 100)
    footprint = SimpleNamespace(x0=0, y0=0, x1=2, y1=3)
    return model, data, EvacuationTeam(model, data, count, footprint)


def test_patient_moves_outdoors_under_forces_without_position_overwrites():
    model, data, team = _scene()
    team.begin(data, 0, 0, np.array([1.8, 1, 0.14]), 0)
    initial_qpos = data.qpos.copy()
    for step in range(6000):
        patient = team.positions(data)[0]
        robot = np.array([[max(-1.0, patient[0] - 0.45), 1, 0.14]])
        team.control(data, robot)
        mujoco.mj_step(model, data)
        team.observe(data, (step + 1) * 0.002)
        if team.done:
            break
    assert team.contacted == team.rescued == 1 and team.done
    assert team.positions(data)[0, 0] <= -0.5
    assert team.distance_m[0] > 2.3 and not np.array_equal(initial_qpos, data.qpos)
    assert team.trail_high_water < 32


def test_wall_contacts_prevent_a_false_rescue():
    model, data, team = _scene(wall=True)
    team.begin(data, 0, 0, np.array([1.8, 1, 0.14]), 0)
    for step in range(5000):
        team.control(data, np.array([[-1, 1, 0.14]]))
        mujoco.mj_step(model, data)
        team.observe(data, (step + 1) * 0.002)
    assert team.positions(data)[0, 0] > 1.0
    assert not team.done and team.rescued == 0


def test_waiting_survivor_never_counts_as_rescued_even_if_outside():
    _model, data, team = _scene()
    team.footprint = SimpleNamespace(x0=3, y0=0, x1=4, y1=3)
    team.observe(data, 1)
    assert team.rescued == 0
    before = data.qpos.copy()
    team.control(data, np.array([[3, 1, 0.14]]))
    np.testing.assert_array_equal(before, data.qpos)
    assert not np.any(data.xfrc_applied)


def test_contact_requires_proximity_and_unique_ownership():
    _model, data, team = _scene(count=2)
    with pytest.raises(ValueError, match="contact"):
        team.begin(data, 0, 0, np.array([4, 1, 0.14]), 0)
    team.begin(data, 0, 0, np.array([1.8, 1, 0.14]), 1)
    with pytest.raises(ValueError, match="already has"):
        team.begin(data, 0, 1, np.array([1.8, 1, 0.14]), 2)
    with pytest.raises(ValueError, match="already owns"):
        team.begin(data, 1, 0, np.array([1.8, 2, 0.14]), 2)


def test_local_detection_requires_range_and_structural_visibility():
    model, data, team = _scene(wall=True)
    rng = np.random.default_rng(7)
    assert team.sense(model, data, 0, np.array([0, 1, 0.14]), rng) is None
    assert team.sense(model, data, 0, np.array([8, 1, 0.14]), rng) is None
    assert team.sense(model, data, 0, np.array([1.5, 1, 0.14]), rng) is not None


def test_escort_waits_for_lagging_patient_and_trail_is_bounded():
    _model, data, team = _scene()
    team.config = replace(team.config, maximum_trail_points=2)
    team.begin(data, 0, 0, np.array([1.8, 1, 0.14]), 0)
    assert team.control(data, np.array([[2.7, 1, 0.14]]))[0]
    team.control(data, np.array([[2.9, 1, 0.14]]))
    with pytest.raises(RuntimeError, match="capacity"):
        team.control(data, np.array([[3.1, 1, 0.14]]))


def test_exterior_routing_ignores_unreachable_preferred_exit():
    from cascade.extract import TraversabilityGraph, VoxelGrid

    cells = np.array([[i, 0, 0] for i in range(5)])
    empty = np.zeros((5, 1, 1), bool)
    graph = TraversabilityGraph(
        VoxelGrid((-2.5, 0, 0), (1, 1, 1), empty, empty),
        cells,
        np.zeros(5),
        np.array([[2, 3], [3, 4]]),
        np.ones(2),
    )
    footprint = SimpleNamespace(x0=-0.1, y0=0, x1=0.1, y1=1)
    goal, path = exterior_route(
        graph, np.array([0, 0.5, 0]), footprint, preferred=np.array([-2, 0.5, 0])
    )
    assert goal[0] == 2 and path[-1, 0] == 2
    isolated = replace(graph, edges=np.empty((0, 2), np.int32), edge_cost=np.empty(0))
    with pytest.raises(NoEvacuationRouteError):
        exterior_route(isolated, np.array([0, 0.5, 0]), footprint)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mass_kg": 0},
        {"speed_m_s": float("nan")},
        {"sensor_noise_m": -1},
        {"maximum_trail_points": 1},
        {"maximum_trail_points": 2.5},
    ],
)
def test_invalid_survivor_models_rejected(overrides):
    with pytest.raises(ValueError):
        EvacuationConfig(**overrides)
