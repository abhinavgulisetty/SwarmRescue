"""BLE survivor-localization contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ember.localization import (
    LocalizationEstimate,
    RSSIConfig,
    localize_rssi,
    rssi_to_range,
    sample_rssi,
)


@pytest.mark.parametrize(
    "overrides",
    (
        {"reference_distance_m": 0.0},
        {"path_loss_exponent": 0.0},
        {"wall_loss_db": -1.0},
        {"noise_std_db": -1.0},
        {"minimum_distance_m": 0.0},
        {"reference_rssi_dbm": np.inf},
    ),
)
def test_rssi_config_rejects_invalid_values(overrides) -> None:
    with pytest.raises(ValueError):
        replace(RSSIConfig(), **overrides)


def test_rssi_range_round_trip_including_walls() -> None:
    config = RSSIConfig(noise_std_db=0.0)
    robots = np.asarray(((0.0, 0.0), (4.0, 0.0), (0.0, 3.0)))
    survivor = np.asarray((1.0, 1.0))
    walls = np.asarray((0.0, 1.0, 2.0))

    rssi = sample_rssi(
        robots,
        survivor,
        np.random.default_rng(7),
        wall_counts=walls,
        config=config,
    )

    np.testing.assert_allclose(
        rssi_to_range(rssi, wall_counts=walls, config=config),
        np.linalg.norm(robots - survivor, axis=1),
    )


def test_four_robots_localize_survivor_from_noisy_rssi() -> None:
    robots = np.asarray(((1.0, 1.0), (9.0, 1.0), (1.0, 8.0), (9.0, 8.0)))
    survivor = np.asarray((6.0, 4.0))
    config = RSSIConfig(noise_std_db=0.35)
    rssi = sample_rssi(
        robots,
        survivor,
        np.random.default_rng(11),
        config=config,
    )

    estimate = localize_rssi(robots, rssi, config=config)

    assert np.linalg.norm(estimate.position_m - survivor) < 0.5
    assert estimate.uncertainty_radius_m > 0.0
    assert estimate.residual_rms_m < 0.5


def test_storey_aware_localization_uses_three_dimensions() -> None:
    robots = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (10.0, 10.0, 0.0),
            (0.0, 0.0, 6.0),
            (10.0, 0.0, 6.0),
            (0.0, 10.0, 6.0),
            (10.0, 10.0, 6.0),
        )
    )
    survivor = np.asarray((4.0, 6.0, 3.5))
    config = RSSIConfig(noise_std_db=0.20)
    rssi = sample_rssi(
        robots,
        survivor,
        np.random.default_rng(13),
        config=config,
    )

    estimate = localize_rssi(robots, rssi, config=config)

    assert estimate.dimension == 3
    assert estimate.covariance_m2.shape == (3, 3)
    assert np.linalg.norm(estimate.position_m - survivor) < 1.0


def test_coverage_uses_the_ellipse_not_its_major_axis_circle() -> None:
    estimate = LocalizationEstimate(
        position_m=np.asarray((0.0, 1.0)),
        covariance_m2=np.diag((1.0, 0.01)),
        residual_rms_m=0.0,
        measurements=3,
    )
    truth = np.zeros(2)

    assert np.linalg.norm(estimate.position_m - truth) < estimate.uncertainty_radius_m
    assert estimate.mahalanobis_squared(truth) == pytest.approx(100.0)
    assert not estimate.covers_95(truth)


@pytest.mark.parametrize(
    "covariance",
    (
        np.asarray(((1.0, 0.5), (0.0, 1.0))),
        np.diag((1.0, 0.0)),
        np.diag((1.0, -1.0)),
    ),
)
def test_localization_rejects_non_spd_covariance(covariance) -> None:
    with pytest.raises(ValueError, match="symmetric positive definite"):
        LocalizationEstimate(
            position_m=np.zeros(2),
            covariance_m2=covariance,
            residual_rms_m=0.0,
            measurements=3,
        )


def test_localization_requires_three_robots() -> None:
    with pytest.raises(ValueError, match="dimension \\+ 1 robots"):
        localize_rssi(np.zeros((2, 2)), np.asarray((-50.0, -55.0)))
