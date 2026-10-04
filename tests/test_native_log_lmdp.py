"""The host adapter must preserve entire integer solves, including capped runs."""

import shutil
from dataclasses import replace

import numpy as np
import pytest

from ember.routing import LMDPConfig, LMDPModel
from mcu.log_lmdp import solve_log_q16
from mcu.native_log_lmdp import solve_native_log_q16


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
