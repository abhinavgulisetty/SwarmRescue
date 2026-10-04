"""Bit-exact deadline fallback and native proposal certification."""

import ctypes
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cascade.extract import TraversabilityGraph, VoxelGrid
from ember.routing.deadline import deadline_route


class Label(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int32) for name in ("node", "time", "cost", "previous")]
    _fields_ += [("state", ctypes.c_uint8)]


class Result(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int32) for name in ("travel_steps", "route_length")]
    _fields_ += [("objective_q16", ctypes.c_int64)]
    _fields_ += [
        (name, ctypes.c_int32)
        for name in ("feasible", "optimal", "resource_fallback", "labels_used")
    ]


def test_native_pareto_solver_and_certificate_match_integer_reference(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler unavailable")
    root = Path(__file__).resolve().parents[1]
    library = tmp_path / "deadline.so"
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
            str(root / "firmware/deadline.c"),
            "-o",
            str(library),
        ],
        check=True,
    )
    native = ctypes.CDLL(str(library))
    i32 = ctypes.c_int32
    ptr = ctypes.POINTER(i32)
    byte_ptr = ctypes.POINTER(ctypes.c_uint8)
    native.ember_deadline_route.argtypes = (
        [i32] * 2
        + [ptr] * 4
        + [byte_ptr]
        + [i32] * 3
        + [
            ctypes.POINTER(Label),
            ptr,
            ctypes.POINTER(ctypes.c_int64),
            ptr,
            byte_ptr,
            ptr,
            ctypes.POINTER(Result),
        ]
    )
    native.ember_deadline_accept.argtypes = (
        [i32] * 2 + [ptr] * 3 + [byte_ptr] + [i32] * 2 + [ptr, i32]
    )

    def pointer(values):
        return values.ctypes.data_as(
            {
                np.dtype(np.int32): ptr,
                np.dtype(np.int64): ctypes.POINTER(ctypes.c_int64),
                np.dtype(np.uint8): byte_ptr,
            }[values.dtype]
        )

    n = 6
    occupied = np.zeros((n, 1, 1), dtype=bool)
    rng = np.random.default_rng(83)
    edges = np.array([[0, 1], [1, 2], [2, 5], [0, 3], [3, 4], [4, 5], [1, 4]])
    for _ in range(40):
        lengths = rng.uniform(0.4, 2.5, len(edges))
        risk = rng.uniform(0.0, 6.0, n)
        graph = TraversabilityGraph(
            VoxelGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), occupied, occupied),
            np.array([[i, 0, 0] for i in range(n)]),
            np.zeros(n),
            edges,
            lengths,
        )
        adjacent = [[] for _ in range(n)]
        for (left, right), length in zip(edges, lengths, strict=True):
            steps = max(1, math.ceil(length / 0.8 / 0.5))
            for source, target in ((left, right), (right, left)):
                cost = round((length + risk[target] * steps * 0.5) * 65536)
                adjacent[source].append((target, steps, cost))
        rows = np.array([0, *np.cumsum([len(row) for row in adjacent])], dtype=np.int32)
        flat = np.array([edge for row in adjacent for edge in row], dtype=np.int32)
        columns, times, costs = (flat[:, i].copy() for i in range(3))
        exits = np.array([0, 0, 0, 0, 0, 1], dtype=np.uint8)
        for capacity in (1, 2, 64):
            horizon = int(rng.integers(1, 40))
            expected = deadline_route(
                graph,
                0,
                (5,),
                risk,
                available_steps=horizon,
                walking_speed_m_s=0.8,
                dt_s=0.5,
                max_labels=capacity,
            )
            labels = (Label * capacity)()
            fast_time = np.empty(n, dtype=np.int32)
            fast_cost = np.empty(n, dtype=np.int64)
            previous = np.empty(n, dtype=np.int32)
            visited = np.empty(n, dtype=np.uint8)
            route = np.empty(n, dtype=np.int32)
            result = Result()
            status = native.ember_deadline_route(
                n,
                len(columns),
                pointer(rows),
                pointer(columns),
                pointer(times),
                pointer(costs),
                pointer(exits),
                0,
                horizon,
                capacity,
                labels,
                pointer(fast_time),
                pointer(fast_cost),
                pointer(previous),
                pointer(visited),
                pointer(route),
                ctypes.byref(result),
            )
            assert status == 0
            assert tuple(route[: result.route_length]) == expected.route
            assert result.travel_steps == expected.travel_steps
            assert result.objective_q16 / 65536 == expected.estimated_objective
            assert bool(result.feasible) == expected.feasible
            assert bool(result.optimal) == expected.optimal
            assert bool(result.resource_fallback) == expected.resource_fallback
            assert result.labels_used == expected.labels_used
            accepted = native.ember_deadline_accept(
                n,
                len(columns),
                pointer(rows),
                pointer(columns),
                pointer(times),
                pointer(exits),
                0,
                horizon,
                pointer(route),
                result.route_length,
            )
            assert bool(accepted) == expected.feasible
