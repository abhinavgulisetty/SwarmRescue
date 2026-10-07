"""Native host execution of the same C99 integer kernel used in conformance tests.

This requires a local C compiler. It changes host execution speed, not the
arithmetic, modeled cycles, or the claim boundary around embedded timing.
"""

import ctypes
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from mcu import MCUArena
from mcu.log_lmdp import LogQ16Model, LogQ16Solution, _array

STRATEGIES = {
    "jacobi": 0,
    "gauss_seidel": 1,
    "bound_jacobi": 2,
    "bound_ordered": 3,
    "prioritized": 4,
    "bound_prioritized": 5,
    "bound_diagonal": 6,
    "auto": 7,
}


class _Stats(ctypes.Structure):
    _fields_ = [
        ("backups", ctypes.c_uint64),
        ("bound_edges", ctypes.c_uint64),
        ("sweeps", ctypes.c_uint32),
        ("fixed_point", ctypes.c_uint32),
        ("strategy", ctypes.c_uint32),
    ]


@dataclass(frozen=True, slots=True)
class NativeLogStats:
    backups: int
    bound_edges: int
    sweeps: int
    fixed_point: bool
    strategy: str


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
    library.ember_log_q16_workspace.argtypes = [ctypes.c_int32] * 3
    library.ember_log_q16_workspace.restype = ctypes.c_int32
    library.ember_log_q16_solve.argtypes = (
        [ctypes.c_int32] * 2
        + [pointer] * 4
        + [byte_pointer] * 2
        + [ctypes.c_int32] * 2
        + [pointer] * 2
        + [ctypes.c_int32, ctypes.POINTER(_Stats)]
    )
    library.ember_log_q16_solve.restype = ctypes.c_int
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


def native_workspace_words(n, edges, strategy="jacobi"):
    if strategy not in STRATEGIES:
        raise ValueError("unknown native integer solve strategy")
    if not 1 <= n <= 2**31 - 1 or not 0 <= edges <= 2**31 - 1:
        raise ValueError("invalid native model size")
    words = _library().ember_log_q16_workspace(n, edges, STRATEGIES[strategy])
    if words < 0:
        raise ValueError("native workspace size exceeds its finite range")
    return words


def native_log_q16_solve(model, values, workspace, *, limit, strategy="jacobi"):
    """Whole C solve; caller-owned buffers include preprocessing scratch.

    Jacobi preserves legacy capped-run semantics. Other schedules can return
    different intermediate fields; compare their converged fixed points.
    """
    words = native_workspace_words(model.n_states, model.col_idx.size, strategy)
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit < 2**31:
        raise ValueError("native sweep cap must be in 1..INT32_MAX")
    for array, size in ((values, model.n_states), (workspace, words)):
        if (
            array.ndim != 1
            or array.size < size
            or array.dtype != np.int32
            or not array.flags.c_contiguous
            or not array.flags.writeable
        ):
            raise ValueError("native output and scratch require writable int32 buffers")
    if values.size != model.n_states or np.shares_memory(values, workspace):
        raise ValueError("native output and scratch must have distinct declared sizes")
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
        or np.shares_memory(array, values)
        or np.shares_memory(array, workspace)
        for array, size, dtype in arrays
    ):
        raise ValueError(
            "native model arrays must be contiguous and separate from scratch"
        )

    def pointer(array):
        kind = ctypes.c_uint8 if array.dtype == np.uint8 else ctypes.c_int32
        return array.ctypes.data_as(ctypes.POINTER(kind))

    stats = _Stats()
    status = _library().ember_log_q16_solve(
        model.n_states,
        model.col_idx.size,
        *(pointer(a) for a, _, _ in arrays),
        limit,
        STRATEGIES[strategy],
        pointer(values),
        pointer(workspace),
        workspace.size,
        ctypes.byref(stats),
    )
    if status:
        raise RuntimeError(f"native integer routing rejected its model: {status}")
    return NativeLogStats(
        stats.backups,
        stats.bound_edges,
        stats.sweeps,
        bool(stats.fixed_point),
        next(name for name, code in STRATEGIES.items() if code == stats.strategy),
    )


def solve_native_log_q16(model, *, arena=None, strategy="jacobi"):
    """Quantize and solve using a selected bounded-memory C schedule."""
    storage = arena if arena is not None else MCUArena(200 * 1024)
    quantized = LogQ16Model.from_float(model, storage)
    values = _array(storage, np.empty(model.n_states, dtype=np.int32), np.int32)
    words = native_workspace_words(model.n_states, model.col_idx.size, strategy)
    workspace = _array(storage, np.empty(words, dtype=np.int32), np.int32)
    stats = native_log_q16_solve(
        quantized, values, workspace, limit=model.config.max_sweeps, strategy=strategy
    )
    return LogQ16Solution(
        quantized, values, stats.sweeps, stats.fixed_point, storage.high_water
    )
