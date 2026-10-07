"""The host adapter must preserve entire integer solves, including capped runs."""

import shutil
from dataclasses import replace

import numpy as np
import pytest

from ember.routing import LMDPConfig, LMDPModel
from mcu import INT32_MAX
from mcu.log_lmdp import LogQ16Model, log_q16_sweep, solve_log_q16
from mcu.native_log_lmdp import (
    STRATEGIES,
    native_log_q16_solve,
    native_workspace_words,
    solve_native_log_q16,
)


@pytest.mark.skipif(shutil.which("cc") is None, reason="C compiler unavailable")
def test_native_complete_and_capped_solves_match_python():
    rng = np.random.default_rng(829)
    for _ in range(12):
        n = 8
        probabilities = rng.uniform(0.05, 1.0, (n, n))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        model = LMDPModel(
            np.arange(n + 1) * n,
            np.tile(np.arange(n), n),
            probabilities.ravel(),
            rng.uniform(0.2, 3.0, n),
            np.arange(n) == n - 1,
            np.arange(n) == n - 2,
            LMDPConfig(max_sweeps=1000),
        )
        for cap in (1, 1000):
            current = replace(model, config=replace(model.config, max_sweeps=cap))
            reference = solve_log_q16(current)
            native = solve_native_log_q16(current)
            np.testing.assert_array_equal(
                native.negative_log_desirability,
                reference.negative_log_desirability,
            )
            assert native.sweeps == reference.sweeps
            assert native.fixed_point == reference.fixed_point
            assert native.arena_high_water_bytes == reference.arena_high_water_bytes
            assert native.route(0) == reference.route(0)


@pytest.mark.skipif(shutil.which("cc") is None, reason="C compiler unavailable")
def test_all_schedules_reach_the_canonical_integer_fixed_point():
    rng = np.random.default_rng(18391)
    for trial in range(60):
        n = int(rng.integers(4, 22))
        probabilities = rng.uniform(0.01, 1.0, (n, n))
        if trial % 2:
            probabilities[rng.random((n, n)) < 0.8] = 0.0
            probabilities[np.arange(n), np.arange(n)] += 0.1
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        model = LMDPModel(
            np.arange(n + 1) * n,
            np.tile(np.arange(n), n),
            probabilities.ravel(),
            rng.uniform(0.02, 2.0, n),
            np.arange(n) >= n - 2,
            np.arange(n) == n - 3,
            LMDPConfig(max_sweeps=10000),
        )
        reference = solve_native_log_q16(model)
        assert reference.fixed_point
        for strategy in STRATEGIES:
            result = solve_native_log_q16(model, strategy=strategy)
            assert result.fixed_point, (trial, strategy)
            np.testing.assert_array_equal(
                result.negative_log_desirability, reference.negative_log_desirability
            )
            for start in range(n):
                assert result.route(start) == reference.route(start)


def _saturated_parallel_model():
    # Each individual path overflows; their soft sum remains finite. Bound=MAX
    # states must still be updated, and the Dijkstra bound must never wrap.
    return LogQ16Model(
        np.array([0, 2, 3], np.int32),
        np.array([1, 1, 1], np.int32),
        np.array([45426, 45426, 0], np.int32),
        np.array([INT32_MAX - 20000, 0], np.int32),
        np.array([0, 1], np.uint8),
        np.array([0, 0], np.uint8),
    )


def test_saturated_path_bounds_do_not_discard_finite_soft_paths():
    model = _saturated_parallel_model()
    current = np.array([INT32_MAX, 0], np.int32)
    expected = np.empty_like(current)
    log_q16_sweep(model, current, expected)
    assert expected[0] == INT32_MAX - 20000
    for strategy in STRATEGIES:
        values = np.empty(2, np.int32)
        workspace = np.empty(native_workspace_words(2, 3, strategy), np.int32)
        stats = native_log_q16_solve(
            model, values, workspace, limit=10, strategy=strategy
        )
        assert stats.fixed_point
        np.testing.assert_array_equal(values, expected)


def test_native_whole_solver_rejects_unsafe_buffers_and_invalid_models():
    model = _saturated_parallel_model()
    values = np.empty(2, np.int32)
    words = native_workspace_words(2, 3, "bound_ordered")
    workspace = np.empty(words, np.int32)
    for invalid in (workspace[:-1], workspace.astype(np.int64), workspace[::2]):
        with pytest.raises(ValueError):
            native_log_q16_solve(
                model, values, invalid, limit=10, strategy="bound_ordered"
            )
    with pytest.raises(ValueError):
        native_log_q16_solve(model, values, values, limit=10)
    with pytest.raises(ValueError):
        native_log_q16_solve(model, model.cost_over_temperature, workspace, limit=10)
    for limit in (0, -1, True, 2**31):
        with pytest.raises(ValueError):
            native_log_q16_solve(model, values, workspace, limit=limit)
    for broken in (
        replace(model, col_idx=np.array([2, 1, 1], np.int32)),
        replace(model, row_ptr=np.array([0, 4, 3], np.int32)),
        replace(model, cost_over_temperature=np.array([-1, 0], np.int32)),
        replace(model, death_mask=np.array([0, 1], np.uint8)),
    ):
        with pytest.raises(RuntimeError, match="rejected its model"):
            native_log_q16_solve(broken, values, workspace, limit=10)


def test_cap_does_not_claim_a_bound_field_is_converged():
    model = _saturated_parallel_model()
    values = np.empty(2, np.int32)
    workspace = np.empty(native_workspace_words(2, 3, "bound_ordered"), np.int32)
    stats = native_log_q16_solve(
        model, values, workspace, limit=1, strategy="bound_ordered"
    )
    assert not stats.fixed_point
    assert stats.sweeps == 1


@pytest.mark.parametrize(
    "self_log,exit_log,cost",
    [
        (1, 726817, 1),
        (328, 347230, 655),
        (6905, 0, INT32_MAX - 100),
    ],
)
def test_diagonal_search_preserves_greatest_rounded_fixed_point(
    self_log, exit_log, cost
):
    # Very sticky rows have many neighboring integer fixed points. Searching
    # for any root would silently choose a different field from cold Jacobi.
    model = LogQ16Model(
        np.array([0, 2, 3], np.int32),
        np.array([0, 1, 1], np.int32),
        np.array([self_log, exit_log, 0], np.int32),
        np.array([cost, 0], np.int32),
        np.array([0, 1], np.uint8),
        np.array([0, 0], np.uint8),
    )
    fields = []
    for strategy in ("jacobi", "bound_diagonal", "auto"):
        words = native_workspace_words(2, 3, strategy)
        guarded = np.full(words + 16, 0x55555555, np.int32)
        values = np.empty(2, np.int32)
        stats = native_log_q16_solve(
            model, values, guarded[8:-8], limit=1000000, strategy=strategy
        )
        assert stats.fixed_point
        assert np.all(guarded[:8] == 0x55555555)
        assert np.all(guarded[-8:] == 0x55555555)
        fields.append(values)
    np.testing.assert_array_equal(fields[0], fields[1])
    np.testing.assert_array_equal(fields[0], fields[2])
