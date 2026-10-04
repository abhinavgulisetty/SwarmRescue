"""Cross-language conformance for the Phase-3 Tier-B Q16 kernel."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mcu import Q16_SCALE
from mcu.lmdp import q16_lmdp_sweep


def _pointer(values: np.ndarray, kind):
    return values.ctypes.data_as(ctypes.POINTER(kind))


def test_c_q16_lmdp_sweep_matches_python_bit_exactly(tmp_path: Path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for Tier-B host conformance")
    root = Path(__file__).resolve().parents[1]
    library = tmp_path / "libember_lmdp.so"
    subprocess.run(
        (
            compiler,
            "-std=c99",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-shared",
            "-fPIC",
            str(root / "firmware" / "lmdp_q16.c"),
            "-o",
            str(library),
        ),
        check=True,
    )
    native = ctypes.CDLL(str(library))
    sweep = native.ember_lmdp_q16_sweep
    sweep.restype = ctypes.c_int32

    row_ptr = np.array((0, 2, 4, 5), dtype=np.int32)
    col_idx = np.array((0, 1, 0, 2, 2), dtype=np.int32)
    passive = np.array(
        (
            Q16_SCALE // 4,
            3 * Q16_SCALE // 4,
            Q16_SCALE // 2,
            Q16_SCALE // 2,
            Q16_SCALE,
        ),
        dtype=np.int32,
    )
    attenuation = np.array(
        (int(0.9 * Q16_SCALE), int(0.8 * Q16_SCALE), Q16_SCALE),
        dtype=np.int32,
    )
    current = np.array(
        (int(0.2 * Q16_SCALE), int(0.7 * Q16_SCALE), Q16_SCALE),
        dtype=np.int32,
    )
    exits = np.array((0, 0, 1), dtype=np.uint8)
    deaths = np.zeros(3, dtype=np.uint8)
    expected = q16_lmdp_sweep(
        row_ptr,
        col_idx,
        passive,
        attenuation,
        current,
        exits,
        deaths,
    )
    actual = np.empty(3, dtype=np.int32)

    status = sweep(
        3,
        _pointer(row_ptr, ctypes.c_int32),
        _pointer(col_idx, ctypes.c_int32),
        _pointer(passive, ctypes.c_int32),
        _pointer(attenuation, ctypes.c_int32),
        _pointer(current, ctypes.c_int32),
        _pointer(exits, ctypes.c_uint8),
        _pointer(deaths, ctypes.c_uint8),
        _pointer(actual, ctypes.c_int32),
    )

    assert status == 0
    np.testing.assert_array_equal(actual, expected)
    assert actual[2] == Q16_SCALE
