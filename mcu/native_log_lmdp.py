"""Native host execution of the same C99 integer kernel used in conformance tests.

This requires a local C compiler. It changes host execution speed, not the
arithmetic, modeled cycles, or the claim boundary around embedded timing.
"""

import ctypes
import shutil
import subprocess
import tempfile
from functools import cache
from pathlib import Path

import numpy as np

from mcu.log_lmdp import solve_log_q16


@cache
def _library():
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("native host routing requires a C compiler")
    source = Path(__file__).resolve().parents[1] / "firmware/log_lmdp.c"
    with tempfile.TemporaryDirectory(prefix="ember-native-") as folder:
        path = Path(folder) / "log_lmdp.so"
        subprocess.run(
            [
                compiler,
                "-std=c99",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-shared",
                "-fPIC",
                str(source),
                "-o",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        library = ctypes.CDLL(str(path))
    pointer = ctypes.POINTER(ctypes.c_int32)
    byte_pointer = ctypes.POINTER(ctypes.c_uint8)
    library.ember_log_q16_sweep.argtypes = (
        [ctypes.c_int32] * 2 + [pointer] * 5 + [byte_pointer] * 2 + [pointer]
    )
    library.ember_log_q16_sweep.restype = ctypes.c_int
    return library


def native_log_q16_sweep(model, current, output):
    if (
        current.shape != (model.n_states,)
        or output.shape != current.shape
        or current.dtype != np.int32
        or output.dtype != np.int32
        or np.shares_memory(current, output)
        or not current.flags.c_contiguous
        or not output.flags.c_contiguous
        or not output.flags.writeable
    ):
        raise ValueError(
            "native state buffers must be distinct contiguous int32 vectors"
        )
    arrays = (
        (model.row_ptr, model.n_states + 1, np.int32),
        (model.col_idx, model.col_idx.size, np.int32),
        (model.negative_log_passive, model.col_idx.size, np.int32),
        (model.cost_over_temperature, model.n_states, np.int32),
        (model.exit_mask, model.n_states, np.uint8),
        (model.death_mask, model.n_states, np.uint8),
    )
    if any(
        array.shape != (size,)
        or array.dtype != dtype
        or not array.flags.c_contiguous
        or np.shares_memory(array, output)
        for array, size, dtype in arrays
    ):
        raise ValueError(
            "native model arrays must have their declared sizes and dtypes"
        )

    def pointer(array):
        kind = ctypes.c_uint8 if array.dtype == np.uint8 else ctypes.c_int32
        return array.ctypes.data_as(ctypes.POINTER(kind))

    status = _library().ember_log_q16_sweep(
        model.n_states,
        model.col_idx.size,
        pointer(model.row_ptr),
        pointer(model.col_idx),
        pointer(model.negative_log_passive),
        pointer(model.cost_over_temperature),
        pointer(current),
        pointer(model.exit_mask),
        pointer(model.death_mask),
        pointer(output),
    )
    if status:
        raise RuntimeError(f"native integer routing rejected its model: {status}")


def solve_native_log_q16(model, *, arena=None):
    return solve_log_q16(model, arena=arena, _sweep=native_log_q16_sweep)
