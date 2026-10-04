"""Autonomous survivor population, localization, and assignment checks."""

from __future__ import annotations

import numpy as np

from ember.search import localize_and_assign


def test_cooperative_search_localizes_and_assigns_unique_targets() -> None:
    robots = np.asarray(((0.0, 0.0), (8.0, 0.0), (0.0, 8.0), (8.0, 8.0)))
    survivors = np.asarray(((1.5, 2.0), (6.5, 1.5), (2.0, 6.5), (6.0, 6.0)))

    result = localize_and_assign(robots, survivors, np.random.default_rng(7))

    assert result.max_error_m < 1.0
    assert sorted(result.robot_to_beacon) == [0, 1, 2, 3]
    assert result.rssi_dbm.shape == (4, 4)
    assert result.covariance_m2.shape == (4, 2, 2)
    assert np.all(result.uncertainty_radii_m > 0.0)
