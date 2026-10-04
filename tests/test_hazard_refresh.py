"""Hazard observations remain live without requiring a new geometry epoch."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from cascade.demo import main, run_demo
from cascade.params import CascadeParams
from ember.physical_team import PhysicalRobotTeam
from ember.radio import RadioConfig
from tests.test_physical import _world


@pytest.mark.parametrize("interval", [-1, float("nan"), float("inf")])
def test_invalid_cadence_rejected_before_loading_scene(interval):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        run_demo(
            7,
            CascadeParams(),
            Path("/unused"),
            n_robots=4,
            live_disaster=True,
            distributed_physical=True,
            sensing_interval_s=interval,
        )


def test_periodic_sensing_requires_distributed_team():
    with pytest.raises(ValueError, match="periodic sensing requires"):
        run_demo(
            7,
            CascadeParams(),
            Path("/unused"),
            live_disaster=True,
            sensing_interval_s=2,
        )


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "2"])
def test_cli_rejects_invalid_or_unsupported_cadence(value):
    with pytest.raises(SystemExit) as exc:
        main(["--sensing-interval-s", value, "--live-disaster", "--no-render"])
    assert exc.value.code == 2


def test_stationary_geometry_observations_follow_heating_and_cooling():
    graph, domain = _world()
    positions = graph.world_positions()
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    team = PhysicalRobotTeam(
        graph,
        positions[[0, 4]],
        loss_seed=13,
        latest_evidence=True,
        radio_config=RadioConfig(packet_error_rate=0),
    )
    cold = domain.initial_state()
    hot = cold.ignite(
        domain, tuple(positions[1]), radius_m=0.51, temperature_k=900, smoke_kg_m3=0.25
    )
    means, slots, replaced = [], [], []
    for state in (cold, hot, cold):
        team.observe_and_exchange(
            domain,
            state,
            positions[[0, 4]],
            positions[[4, 0]],
            np.random.default_rng(7),
            model=model,
            data=data,
            scan_radius_m=4,
            beta=2,
        )
        means.append(team.robots[0].engine.belief().mean[1])
        slots.append(len(team.robots[0].seen_factors))
        replaced.append(team.robots[0].replaced_factors)
        assert team.graph is graph and team.map_epoch == 0
        assert team.geometry_revisions == 0
        assert all(r.rejected_epoch_packets == 0 for r in team.robots)
    assert means[1] > means[0] + 1
    assert means[2] < means[1] - 1
    assert slots[0] == slots[1] == slots[2]
    assert replaced[2] > replaced[1] > replaced[0]
