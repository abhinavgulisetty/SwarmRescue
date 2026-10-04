"""Checks for the Phase 1 sensor-to-belief-to-route slice."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ember.belief import GaussianBelief, HazardSensorConfig
from ember.demo import run_demo
from ember.mission import run_mission


@pytest.mark.parametrize(
    "overrides",
    (
        {"temperature_noise_k": -1.0},
        {"temperature_quantization_k": 0.0},
        {"smoke_noise_kg_m3": -1.0},
        {"smoke_quantization_kg_m3": 0.0},
        {"temperature_bias_k": np.inf},
    ),
)
def test_sensor_config_rejects_invalid_calibration(overrides) -> None:
    with pytest.raises(ValueError):
        replace(HazardSensorConfig(), **overrides)


def test_information_update_matches_two_gaussians_and_handles_duplicates() -> None:
    prior = GaussianBelief.uniform(2, mean=0.0, variance=4.0)
    posterior = prior.update(
        np.asarray((0, 0)),
        np.asarray((2.0, 2.0)),
        np.asarray((1.0, 1.0)),
    )

    assert posterior.mean[0] == pytest.approx(16.0 / 9.0)
    assert posterior.variance[0] == pytest.approx(4.0 / 9.0)
    assert posterior.mean[1] == 0.0
    assert posterior.variance[1] == 4.0
    assert prior.mean[0] == 0.0


def test_stale_belief_loses_precision_and_beta_changes_cost_direction() -> None:
    belief = GaussianBelief.uniform(1, mean=0.3, variance=0.04)
    stale = belief.predict(0.05)

    assert stale.variance[0] > belief.variance[0]
    assert stale.routing_cost(1.0)[0] > belief.routing_cost(1.0)[0]
    assert stale.routing_cost(-1.0)[0] < belief.routing_cost(-1.0)[0]


def test_phase1_demo_searches_unknown_then_both_avoid_measured_smoke() -> None:
    result = run_demo(7)
    mission = run_mission(7)

    assert result.succeeded
    assert result.sensed_smoke_kg_m3 > 0.0
    assert result.localization_error_m < 1.0
    assert result.ember_dose < result.shortest_dose
    assert result.flooding_dose < result.shortest_dose
    assert result.centralized_length < result.flooding_length
    assert result.gbp_sweeps == mission.gbp_sweeps
    assert result.radio_deliveries > 0
    assert result.radio_duplicates > 0
    assert result.sweeps > 0
