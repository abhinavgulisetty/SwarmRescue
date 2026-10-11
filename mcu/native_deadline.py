"""Compiled exact positive-cost, single-deadline host routing; no MCU claim."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import tempfile
from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np

from mcu import MCUArena

MODES = {
    "native_dp": 0,
    "native_full_dp": 1,
    "native_window_dp": 2,
    "native_astar": 3,
    "native_exact": 4,
}


class _Stats(ctypes.Structure):
    _fields_ = (
        [(name, ctypes.c_int64) for name in ("cost", "lower_bound", "travel")]
        + [("peak", ctypes.c_uint64)]
        + [
            (name, ctypes.c_uint32)
            for name in (
                "route_length",
                "feasible",
                "optimal",
                "labels",
                "states",
                "runs",
                "method",
                "expanded",
            )
        ]
    )


@cache
def _library():
    compiler = shutil.which("cc")
    if not compiler:
        raise RuntimeError("compiled deadline routing requires a C compiler")
    source = Path(__file__).resolve().parents[1] / "firmware/deadline_exact.c"
    with tempfile.TemporaryDirectory(prefix="ember-deadline-") as folder:
        library_path = Path(folder) / "deadline.so"
        subprocess.run(
            [
                compiler,
                "-std=c99",
                "-O3",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-shared",
                "-fPIC",
                str(source),
                "-o",
                str(library_path),
            ],
            check=True,
            capture_output=True,
        )
        library = ctypes.CDLL(str(library_path))
    u32 = ctypes.POINTER(ctypes.c_uint32)
    i64 = ctypes.POINTER(ctypes.c_int64)
    byte = ctypes.POINTER(ctypes.c_uint8)
    library.ember_deadline_exact.argtypes = (
        [ctypes.c_uint32, u32, u32, u32, i64, u32, u32, u32, i64, byte]
        + [ctypes.c_uint32] * 4
        + [byte, ctypes.c_size_t, ctypes.c_size_t, u32, ctypes.POINTER(_Stats)]
    )
    library.ember_deadline_exact.restype = ctypes.c_int
    return library


def native_graph_search(
    edges,
    steps,
    costs,
    n,
    start,
    goals,
    deadline,
    max_labels,
    *,
    arena,
    strategy,
    fallback,
):
    from ember.routing.deadline import DeadlineRoute

    storage = arena if arena is not None else MCUArena(200 * 1024)
    source = edges.reshape(-1).astype(np.int64, copy=False)
    target = edges[:, ::-1].reshape(-1).astype(np.int64, copy=False)
    if len(source) >= 2**32:
        raise ValueError("native transition indexing exceeds uint32")
    order = np.argsort(source, kind="stable")
    rows = np.empty(n + 1, np.uint32)
    rows[0] = 0
    np.cumsum(np.bincount(source, minlength=n), out=rows[1:])
    targets = np.ascontiguousarray(target[order], dtype=np.uint32)
    duration = np.ascontiguousarray(np.repeat(steps, 2)[order], dtype=np.uint32)
    forward_weight = np.ascontiguousarray(costs[:, ::-1].reshape(-1)[order])
    reverse_weight = np.ascontiguousarray(costs.reshape(-1)[order])
    # Public graphs have both directions of each undirected edge. Topology
    # and clock arrays are identical in the reverse graph; only risk differs.
    forward = (rows, targets, duration, forward_weight)
    backward = (rows, targets, duration, reverse_weight)
    goal_flags = np.zeros(n, np.uint8)
    goal_flags[list(goals)] = 1
    route = np.empty(n, np.uint32)
    stats = _Stats()
    buffer = (ctypes.c_uint8 * storage.capacity).from_buffer(storage._storage)

    def pointers(csr):
        return [
            a.ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint32 if i < 3 else ctypes.c_int64)
            )
            for i, a in enumerate(csr)
        ]

    status = _library().ember_deadline_exact(
        n,
        *pointers(forward),
        *pointers(backward),
        goal_flags.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        start,
        deadline,
        min(max_labels, 2**32 - 1),
        MODES[strategy],
        buffer,
        storage.capacity,
        storage.used,
        route.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.byref(stats),
    )
    storage.high_water = max(storage.high_water, stats.peak)
    if status == -1:
        raise RuntimeError("compiled deadline certificate or witness failed")
    result = DeadlineRoute(
        tuple(int(node) for node in route[: stats.route_length]),
        int(stats.travel),
        stats.cost / 65536,
        bool(stats.feasible),
        bool(stats.optimal),
        not bool(stats.optimal),
        int(stats.labels),
        storage.high_water,
        lower_bound_q16=int(stats.lower_bound),
        search_strategy=strategy,
        expanded_labels=int(stats.expanded),
        preprocessing_runs=int(stats.runs),
        dynamic_states=int(stats.states),
    )

    if status == -2 and (
        strategy in ("native_exact", "native_dp") or not stats.route_length
    ):
        alternative = fallback()
        if alternative.feasible and (
            not result.feasible
            or alternative.estimated_objective <= result.estimated_objective
        ):
            return replace(
                alternative,
                search_strategy="native_memory_fallback",
                labels_used=max(alternative.labels_used, result.labels_used),
                expanded_labels=alternative.expanded_labels + result.expanded_labels,
            )
        if result.feasible:
            return replace(
                result,
                search_strategy="native_memory_fallback",
                resource_fallback=True,
                lower_bound_q16=max(
                    result.lower_bound_q16 or 0, alternative.lower_bound_q16 or 0
                ),
            )
        return replace(alternative, search_strategy="native_memory_fallback")
    return result
