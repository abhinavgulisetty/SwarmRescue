"""Integer policy and native conformance tests beyond the Q16 underflow floor."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ember.routing import LMDPConfig, LMDPModel, greedy_route, solve_jacobi
from mcu import INT32_MAX, Q16_SCALE, MCUArena, MCUOutOfMemory
from mcu.log_lmdp import SOFTPLUS_LUT, log_q16_sweep, softplus_q16, solve_log_q16


def _chain(n: int, cost: float = 2.0) -> LMDPModel:
    return LMDPModel(
        np.arange(n + 1),
        np.minimum(np.arange(n) + 1, n - 1),
        np.ones(n),
        np.full(n, cost),
        np.arange(n) == n - 1,
        np.zeros(n, dtype=bool),
        LMDPConfig(tolerance=1e-10, max_sweeps=n + 10),
    )


def test_integer_route_survives_float_and_linear_q16_underflow() -> None:
    solution = solve_log_q16(_chain(500))
    assert solution.fixed_point
    assert solution.route(0) == tuple(range(500))
    assert solution.negative_log_desirability[0] == 998 * Q16_SCALE
    assert solution.desirability_reference()[0] == 0.0
    assert solution.arena_high_water_bytes < 200 * 1024


def test_policy_agrees_with_float_far_below_linear_q16_resolution() -> None:
    model = _chain(80)
    reference = solve_jacobi(model)
    integer = solve_log_q16(model)
    assert reference.desirability[0] < 1.0 / Q16_SCALE
    assert integer.route(0) == tuple(greedy_route(model, reference, 0))
    np.testing.assert_allclose(
        integer.negative_log_desirability / Q16_SCALE, reference.value, atol=1e-9
    )


def test_log_kernel_preserves_unreachable_and_death_boundaries() -> None:
    model = LMDPModel(
        np.arange(5), np.array([1, 1, 3, 3]), np.ones(4), np.ones(4),
        np.array([0, 1, 0, 0], dtype=bool),
        np.array([0, 0, 0, 1], dtype=bool), LMDPConfig(),
    )
    solution = solve_log_q16(model)
    assert solution.fixed_point
    assert solution.route(0) == (0, 1)
    assert solution.route(2) == solution.route(3) == ()
    assert solution.negative_log_desirability[2] == INT32_MAX
    with pytest.raises(ValueError, match="alias"):
        log_q16_sweep(
            solution.model, solution.negative_log_desirability,
            solution.negative_log_desirability,
        )
    with pytest.raises(MCUOutOfMemory):
        solve_log_q16(model, arena=MCUArena(8))


def test_softplus_table_accuracy_and_monotonicity() -> None:
    distances = np.arange(0, 18 * Q16_SCALE, 137)
    actual = np.array([softplus_q16(int(value)) for value in distances])
    expected = np.logaddexp(0.0, -distances / Q16_SCALE) * Q16_SCALE
    assert np.max(np.abs(actual - expected)) < 1.2
    assert np.all(np.diff(actual) <= 0)


def test_native_sweeps_and_policy_are_bit_exact(tmp_path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    root = Path(__file__).resolve().parents[1]
    library = tmp_path / "log_lmdp.so"
    subprocess.run(
        [compiler, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-shared",
         "-fPIC", str(root / "firmware/log_lmdp.c"), "-o", str(library)],
        check=True,
    )
    native = ctypes.CDLL(str(library))
    pointer = ctypes.POINTER(ctypes.c_int32)
    byte_pointer = ctypes.POINTER(ctypes.c_uint8)
    native.ember_log_q16_sweep.argtypes = (
        [ctypes.c_int32] * 2 + [pointer] * 5 + [byte_pointer] * 2 + [pointer]
    )
    native.ember_log_q16_next.argtypes = [ctypes.c_int32] * 2 + [pointer] * 4

    def ptr(values):
        return values.ctypes.data_as(
            byte_pointer if values.dtype == np.uint8 else pointer
        )

    # Random normalized CSR rows exercise soft-min interpolation and cancellation.
    rng = np.random.default_rng(77)
    n = 12
    probabilities = rng.uniform(0.01, 1.0, (n, n))
    probabilities /= probabilities.sum(axis=1)[:, None]
    model = LMDPModel(
        np.arange(n + 1) * n, np.tile(np.arange(n), n), probabilities.ravel(),
        rng.uniform(0.2, 4.0, n), np.arange(n) == n - 1,
        np.arange(n) == n - 2, LMDPConfig(),
    )
    solution = solve_log_q16(model)
    quantized = solution.model
    buffers = [quantized.row_ptr, quantized.col_idx,
               quantized.negative_log_passive, quantized.cost_over_temperature]
    for _ in range(30):
        current = rng.integers(0, 100 * Q16_SCALE, n, dtype=np.int32)
        current[rng.random(n) < 0.2] = INT32_MAX
        expected = np.empty_like(current)
        actual = np.empty_like(current)
        log_q16_sweep(quantized, current, expected)
        status = native.ember_log_q16_sweep(
            n, n * n, *(ptr(buffer) for buffer in buffers), ptr(current),
            ptr(quantized.exit_mask), ptr(quantized.death_mask), ptr(actual),
        )
        assert status == 0
        np.testing.assert_array_equal(actual, expected)
    for state in range(n - 2):
        route = solution.route(state)
        actual_next = native.ember_log_q16_next(
            state, n, *(ptr(buffer) for buffer in buffers[:3]),
            ptr(solution.negative_log_desirability),
        )
        assert actual_next == route[1]
    table_text = (root / "firmware/log_lmdp_lut.h").read_text()
    encoded = table_text.split("= {", 1)[1].split("};", 1)[0]
    np.testing.assert_array_equal(
        np.array([int(value) for value in encoded.split(",") if value.strip()]),
        SOFTPLUS_LUT,
    )
