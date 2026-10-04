"""Vector range messages, coupled pose uncertainty and incremental provenance."""

import numpy as np
import pytest

from ember.localization import RSSIConfig, sample_rssi
from ember.range_gbp import RangeFactorGBP, RangeGBPConfig


def engine(beacons=1, variance=0.01, dimension=2, **config):
    anchors = np.array([[0, 0], [8, 0], [0, 8], [8, 8]], dtype=float)
    if dimension == 3:
        anchors = np.array([[0, 0, 0], [8, 0, 0], [0, 8, 0], [0, 0, 8]], dtype=float)
    return RangeFactorGBP(
        anchors,
        np.tile(np.eye(dimension) * variance, (4, 1, 1)),
        beacons,
        config=RangeGBPConfig(**config),
    )


@pytest.mark.parametrize("beacons", [1, 3])
def test_linearized_means_match_joint_information_solve(beacons):
    e = engine(beacons)
    d, n = 2, 4
    seed = np.array([[2 + b, 3 + b] for b in range(beacons)], float)
    difference = seed[None] - e.anchors[:, None]
    distances = np.linalg.norm(difference, axis=2)
    directions = difference / distances[:, :, None]
    offsets = distances + np.arange(4)[:, None] * 0.02
    for source in range(n):
        for b in range(beacons):
            e.observe(source, b, 0, distances[source, b], 0.04)
    e.robot_mean[:] = e.anchors
    e.beacon_mean[:] = seed
    sweeps, converged, pose_cov, beacon_cov = e._linear_solve(directions, offsets, seed)
    assert converged and sweeps < e.config.max_sweeps
    size = (n + beacons) * d
    precision, information = np.zeros((size, size)), np.zeros(size)
    for source in range(n):
        block = slice(d * source, d * (source + 1))
        precision[block, block] += e.pose_precision[source]
        information[block] += e.pose_information[source]
    for b in range(beacons):
        block = slice(d * (n + b), d * (n + b + 1))
        precision[block, block] += np.eye(d) / e.config.beacon_prior_variance_m2
        information[block] += seed[b] / e.config.beacon_prior_variance_m2
        for source in range(n):
            row = np.zeros(size)
            row[d * source : d * (source + 1)] = -directions[source, b]
            row[block] = directions[source, b]
            precision += np.outer(row, row) / e.variances[source, b]
            information += row * offsets[source, b] / e.variances[source, b]
    exact = np.linalg.solve(precision, information).reshape(n + beacons, d)
    np.testing.assert_allclose(e.robot_mean, exact[:n], atol=1e-6)
    np.testing.assert_allclose(e.beacon_mean, exact[n:], atol=1e-6)
    if beacons == 1:  # tree marginals are exact; loopy GBP variances need not be.
        covariance = np.linalg.inv(precision)
        for source in range(n):
            block = slice(d * source, d * (source + 1))
            np.testing.assert_allclose(
                pose_cov[source], covariance[block, block], atol=1e-7
            )
        np.testing.assert_allclose(beacon_cov[0], covariance[-d:, -d:], atol=1e-7)


@pytest.mark.parametrize("dimension", [2, 3])
def test_noiseless_ranges_recover_beacon_without_truth_initializer(dimension):
    e = engine(dimension=dimension)
    truth = np.array([2, 3, 4][:dimension], float)
    for source, distance in enumerate(np.linalg.norm(e.anchors - truth, axis=1)):
        e.observe(source, 0, 0, float(distance), 0.0025)
    result = e.solve()
    assert result.converged
    np.testing.assert_allclose(result.estimates[0].position_m, truth, atol=1e-5)
    assert result.retained_numeric_bytes == e.retained_numeric_bytes
    assert result.retained_numeric_bytes < 20000


def test_pose_uncertainty_inflates_beacon_covariance():
    results = []
    for variance in (1e-6, 0.25):
        e = engine(variance=variance)
        for i, distance in enumerate(np.linalg.norm(e.anchors - [2, 3], axis=1)):
            e.observe(i, 0, 0, float(distance), 0.0025)
        r = e.solve()
        assert r.converged
        results.append(np.trace(r.estimates[0].covariance_m2))
    assert results[1] > 20 * results[0]


def test_latest_factor_rejects_replay_and_moves_estimate_without_slot_growth():
    e = engine()
    for sequence, truth in ((0, [2, 3]), (1, [3, 4])):
        for i, distance in enumerate(np.linalg.norm(e.anchors - truth, axis=1)):
            assert e.observe(i, 0, sequence, float(distance), 0.0025)
        result = e.solve()
        assert result.converged
        np.testing.assert_allclose(result.estimates[0].position_m, truth, atol=1e-4)
    before = e.ranges.copy()
    size = e.retained_numeric_bytes
    assert not e.observe(0, 0, 0, 100, 0.1)
    assert not e.observe(0, 0, 1, 100, 0.1)
    np.testing.assert_array_equal(e.ranges, before)
    assert e.retained_numeric_bytes == size and e.replaced == 4 and e.stale == 2


def test_shared_pose_moves_under_multiple_beacon_factors():
    e = engine(beacons=2, variance=0.1)
    for b, truth in enumerate(([2, 3], [6, 5])):
        for i, distance in enumerate(np.linalg.norm(e.anchors - truth, axis=1)):
            e.observe(i, b, 0, float(distance + 0.2 if i == 0 else distance), 0.0025)
    result = e.solve()
    assert result.converged
    assert np.linalg.norm(result.robot_positions_m[0] - e.anchors[0]) > 0.05
    assert all(np.linalg.eigvalsh(x).min() > 0 for x in result.robot_covariance_m2)


def test_iteration_cap_reports_nonconvergence():
    e = engine(max_sweeps=1, max_linearizations=1)
    truth = np.array([2, 3])
    for i, distance in enumerate(np.linalg.norm(e.anchors - truth, axis=1)):
        e.observe(i, 0, 0, float(distance + 0.3), 0.0025)
    assert not e.solve().converged


def test_incomplete_and_degenerate_anchor_geometry_fail_explicitly():
    e = engine()
    e.observe(0, 0, 0, 2, 0.1)
    with pytest.raises(ValueError, match="received ranges"):
        e.solve()
    e.anchors[:, 1] = 0
    for i in range(4):
        e.observe(i, 0, 1, 2 + i, 0.1)
    with pytest.raises(ValueError, match="geometry"):
        e.solve()


@pytest.mark.parametrize(
    "args",
    [
        (-1, 0, 0, 1, 0.1),
        (0, 2, 0, 1, 0.1),
        (0, 0, -1, 1, 0.1),
        (0, 0, True, 1, 0.1),
        (0, 0, 0, 0, 0.1),
        (0, 0, 0, 1, 0),
        (0, 0, 0, float("nan"), 0.1),
        (0, 0, 2**63, 1, 0.1),
    ],
)
def test_invalid_observations_leave_table_unchanged(args):
    e = engine()
    before = e.sequences.copy()
    with pytest.raises(ValueError):
        e.observe(*args)
    np.testing.assert_array_equal(e.sequences, before)


@pytest.mark.parametrize(
    "config",
    [
        {"damping": 0},
        {"tolerance": float("nan")},
        {"max_sweeps": 0},
        {"max_linearizations": 1.5},
        {"beacon_prior_variance_m2": -1},
    ],
)
def test_invalid_configuration(config):
    with pytest.raises(ValueError):
        RangeGBPConfig(**config)


def test_rssi_path_follows_identical_observations():
    e = engine()
    truth = np.array([2, 3])
    cfg = RSSIConfig(noise_std_db=0)
    rssi = sample_rssi(e.anchors, truth, np.random.default_rng(7), config=cfg)
    for i, value in enumerate(rssi):
        e.observe_rssi(i, 0, 0, float(value), config=cfg)
    result = e.solve()
    assert result.converged
    np.testing.assert_allclose(result.estimates[0].position_m, truth, atol=1e-5)


def test_search_uses_identical_rssi_for_both_localizers_and_unique_assignments():
    from ember.search import localize_and_assign

    robots = engine().anchors
    survivors = np.array([[1.5, 2], [6.5, 1.5], [2, 6.5], [6, 6]], float)
    old = localize_and_assign(robots, survivors, np.random.default_rng(7))
    new = localize_and_assign(
        robots, survivors, np.random.default_rng(7), range_gbp=True
    )
    np.testing.assert_array_equal(old.rssi_dbm, new.rssi_dbm)
    assert new.range_gbp_converged and new.range_gbp_sweeps > 0
    assert new.max_error_m < 1 and sorted(new.robot_to_beacon) == list(range(4))
    assert new.localization_retained_numeric_bytes < 20000


def test_demo_rejects_gbp_without_distributed_mode_before_scene_load():
    from pathlib import Path

    from cascade.demo import main, run_demo
    from cascade.params import CascadeParams

    with pytest.raises(ValueError, match="range GBP requires"):
        run_demo(
            7, CascadeParams(), Path("/unused"), live_disaster=True, range_gbp=True
        )
    with pytest.raises(SystemExit) as exc:
        main(["--range-gbp", "--live-disaster", "--no-render"])
    assert exc.value.code == 2


def test_gbp_search_does_not_call_gauss_newton(monkeypatch):
    import ember.search as search

    def forbidden(*args, **kwargs):
        raise AssertionError("centralized Gauss-Newton was called")

    monkeypatch.setattr(search, "localize_rssi", forbidden)
    robots = engine().anchors
    survivors = np.array([[1.5, 2], [6.5, 1.5], [2, 6.5], [6, 6]], float)
    result = search.localize_and_assign(
        robots, survivors, np.random.default_rng(7), range_gbp=True
    )
    assert result.range_gbp_converged


def test_rssi_replay_does_not_overwrite_log_likelihood():
    e = engine()
    assert e.observe_rssi(0, 0, 3, -60)
    before = e.log_range_variances.copy()
    assert not e.observe_rssi(0, 0, 2, -80)
    np.testing.assert_array_equal(e.log_range_variances, before)
    assert e.observe(0, 0, 4, 2, 0.1)
    assert e.log_range_variances[0, 0] == 0


def test_log_range_map_matches_independent_nonlinear_reference():
    from scipy.optimize import least_squares

    e = engine(variance=0.02)
    truth = np.array([2, 3])
    cfg = RSSIConfig(noise_std_db=1.5)
    rssi = sample_rssi(e.anchors, truth, np.random.default_rng(11), config=cfg)
    for source, reading in enumerate(rssi):
        e.observe_rssi(source, 0, 0, float(reading), config=cfg)
    e._initialize()
    seed = e.beacon_mean.copy()
    initial = np.concatenate((e.anchors.ravel(), seed.ravel()))

    def residual(values):
        poses, beacon = values[:8].reshape(4, 2), values[8:]
        lengths = np.linalg.norm(beacon - poses, axis=1)
        return np.concatenate(
            (
                (poses - e.anchors).ravel() / np.sqrt(0.02),
                np.log(lengths / e.ranges[:, 0]) / np.sqrt(e.log_range_variances[:, 0]),
                (beacon - seed[0]) / np.sqrt(e.config.beacon_prior_variance_m2),
            )
        )

    reference = least_squares(residual, initial, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    result = e.solve()
    assert result.converged and reference.success
    np.testing.assert_allclose(
        result.estimates[0].position_m, reference.x[8:], atol=1e-4
    )
    assert result.objective == pytest.approx(np.sum(reference.fun**2), abs=1e-6)
