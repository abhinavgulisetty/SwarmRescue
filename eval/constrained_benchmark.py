"""Matched deadline-search latency, certificates and quality under label caps."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from cascade.extract import TraversabilityGraph, VoxelGrid
from cascade.floorplan import build_floorplan
from cascade.params import BuildingParams
from cascade.rng import stage_rng
from ember.routing.deadline import deadline_route
from eval.procedural import _exit_nodes, _room_graph, _with_seeded_blockage
from mcu import MCUArena

FAMILIES = ("grid", "rooms", "dense", "deadends", "tradeoff", "large_grid")
METHODS = (
    "pareto",
    "astar",
    "lazy_astar",
    "dual_astar",
    "bounded_dp",
    "window_dp",
    "hybrid_exact",
    "milp",
)
DEVELOPMENT_SEEDS = tuple(range(52000, 52004))
EVALUATION_SEEDS = tuple(range(62000, 62008))
LABEL_CAPS = (32, 128, 4096)


def _graph(n, edges, lengths, positions=None):
    occupied = np.zeros((n, 1, 1), bool)
    cells = np.array([[i, 0, 0] for i in range(n)]) if positions is None else positions
    return TraversabilityGraph(
        VoxelGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), occupied, occupied),
        cells,
        np.zeros(n),
        np.array(edges, np.int64).reshape(-1, 2),
        np.array(lengths, float),
    )


def arcs(graph, risk):
    result = [[] for _ in range(graph.n_nodes)]
    for (a, b), length in zip(graph.edges, graph.edge_cost, strict=True):
        steps = max(1, math.ceil(float(length)))
        for source, target in ((int(a), int(b)), (int(b), int(a))):
            cost = max(1, round((float(length) + risk[target] * steps) * 65536))
            result[source].append((target, steps, cost))
    return result


def _pair_distance(adjacent, start, goals, time_first):
    distance = {start: (0, 0)}
    queue = [(0, 0, start)]
    while queue:
        first, second, node = heapq.heappop(queue)
        if (first, second) != distance[node]:
            continue
        if node in goals:
            return (first, second)
        for target, steps, cost in adjacent[node]:
            candidate = (
                (first + steps, second + cost)
                if time_first
                else (first + cost, second + steps)
            )
            if candidate < distance.get(target, (float("inf"), float("inf"))):
                distance[target] = candidate
                heapq.heappush(queue, (*candidate, target))
    return None


def make_case(seed, family):
    rng = np.random.default_rng(np.random.SeedSequence([seed, FAMILIES.index(family)]))
    if family == "rooms":
        plan = build_floorplan(seed, BuildingParams())
        graph, keys = _room_graph(plan)
        goals = _exit_nodes(plan, graph, keys)
        graph, _ = _with_seeded_blockage(graph, goals, stage_rng(seed, "collapse"))
        positions = graph.world_positions()
        start = int(
            np.argmax(
                np.min(
                    np.linalg.norm(
                        positions[:, None] - positions[np.array(goals)][None], axis=2
                    ),
                    axis=1,
                )
            )
        )
        risk = rng.uniform(0.1, 8, graph.n_nodes)
    elif family == "tradeoff":
        stages = int(rng.integers(10, 18))
        n = 3 * stages + 1
        edges = []
        lengths = []
        risk = np.zeros(n)
        for i in range(stages):
            high = stages + 1 + i
            low = 2 * stages + 1 + i
            edges.extend([(i, high), (high, i + 1), (i, low), (low, i + 1)])
            detour = float(rng.uniform(1.05, 2.95))
            lengths.extend([0.5, 0.5, detour, detour])
            risk[high] = rng.uniform(6, 22)
            risk[low] = rng.uniform(0, 0.5)
        graph = _graph(n, edges, lengths)
        start = 0
        goals = (stages,)
    elif family == "deadends":
        spine = int(rng.integers(25, 45))
        edges = [(i, i + 1) for i in range(spine - 1)]
        lengths = [1.0] * (spine - 1)
        n = spine
        risk = [2.0 + float(rng.random()) for _ in range(spine)]
        for i in range(spine - 1):
            for _ in range(3):
                parent = i
                for _ in range(int(rng.integers(3, 7))):
                    edges.append((parent, n))
                    lengths.append(0.5)
                    risk.append(0.0)
                    parent = n
                    n += 1
        graph = _graph(n, edges, lengths)
        risk = np.array(risk)
        start = 0
        goals = (spine - 1,)
    elif family == "dense":
        n = 64
        edges = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if j == i + 1 or rng.random() < 0.25
        ]
        graph = _graph(n, edges, rng.uniform(0.2, 5, len(edges)))
        risk = rng.uniform(0, 12, n)
        start = 0
        goals = (n - 1,)
    else:
        side = int(rng.choice([32, 40] if family == "large_grid" else [12, 16, 20]))
        n = side * side
        edges = []
        for i in range(n):
            y, x = divmod(i, side)
            if x + 1 < side:
                edges.append((i, i + 1))
            if y + 1 < side:
                edges.append((i, i + side))
        positions = np.array([[i % side, i // side, 0] for i in range(n)])
        graph = _graph(n, edges, np.ones(len(edges)), positions)
        risk = np.full(n, 0.1)
        for _ in range(5):
            center = rng.uniform(0, side, 2)
            sigma = rng.uniform(1, side / 4)
            risk += rng.uniform(6, 16) * np.exp(
                -np.sum((positions[:, :2] - center) ** 2, axis=1) / (2 * sigma * sigma)
            )
        start = 0
        goals = (n - 1,)
        if family == "large_grid":
            # Opposite-wall endpoints force a genuine cost/time tradeoff:
            # avoiding the central hazard band adds traversal time.
            center_y = side // 2
            start = center_y * side
            goals = (start + side - 1,)
            risk += 40 * np.exp(
                -(((positions[:, 0] - side / 2) / 2) ** 2)
                - ((positions[:, 1] - center_y) / (side / 4)) ** 2
            )
    adjacent = arcs(graph, risk)
    fast_time = _pair_distance(adjacent, start, set(goals), True)[0]
    cheap_time = _pair_distance(adjacent, start, set(goals), False)[1]
    slack = (0.0, 0.2, 0.5, 0.9)[seed % 4]
    deadline = fast_time + math.floor(slack * max(0, cheap_time - fast_time))
    if seed % 7 == 0:
        deadline = max(0, fast_time - 1)
    return graph, start, goals, risk, int(deadline)


def dp_optimum(adjacent, start, goals, deadline):
    inf = np.iinfo(np.int64).max
    table = np.full((deadline + 1, len(adjacent)), inf, np.int64)
    table[0, start] = 0
    for elapsed in range(deadline + 1):
        for node in np.flatnonzero(table[elapsed] != inf):
            current = int(table[elapsed, node])
            for target, steps, cost in adjacent[node]:
                arrival = elapsed + steps
                if arrival <= deadline and current + cost < table[arrival, target]:
                    table[arrival, target] = current + cost
    best = int(np.min(table[:, np.array(goals)]))
    return None if best == inf else best


def milp_route(adjacent, start, goals, deadline):
    n = len(adjacent)
    edges = [(u, v, t, c) for u, row in enumerate(adjacent) for v, t, c in row]
    edges.extend((goal, n, 0, 0) for goal in goals)
    rows = []
    columns = []
    weights = []
    for index, (u, v, steps, _) in enumerate(edges):
        rows.extend((u, v, n + 1))
        columns.extend((index,) * 3)
        weights.extend((1.0, -1.0, float(steps)))
    matrix = coo_matrix((weights, (rows, columns)), shape=(n + 2, len(edges))).tocsc()
    lower = np.zeros(n + 2)
    upper = np.zeros(n + 2)
    lower[start] = upper[start] = 1
    lower[n] = upper[n] = -1
    lower[-1] = -np.inf
    upper[-1] = deadline
    solved = milp(
        np.array([c / 65536 for _, _, _, c in edges]),
        integrality=np.ones(len(edges)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"mip_rel_gap": 0.0, "time_limit": 3.0},
    )
    if solved.status == 2:
        return {
            "feasible": False,
            "optimal": True,
            "cost_q16": None,
            "route": (),
            "status": 2,
        }
    if solved.x is None:
        return {
            "feasible": False,
            "optimal": False,
            "cost_q16": None,
            "route": (),
            "status": int(solved.status),
        }
    selected = [i for i, x in enumerate(solved.x) if x > 0.5]
    following = {edges[i][0]: i for i in selected}
    route = [start]
    cost = elapsed = 0
    while route[-1] != n and len(route) <= n + 1:
        index = following.get(route[-1])
        if index is None:
            break
        _, target, steps, weight = edges[index]
        route.append(target)
        cost += weight
        elapsed += steps
    feasible = route[-1] == n and elapsed <= deadline and len(set(route)) == len(route)
    if feasible:
        route = route[:-1]
    optimal = bool(
        solved.success and feasible and cost == sum(edges[i][3] for i in selected)
    )
    return {
        "feasible": feasible,
        "optimal": optimal,
        "cost_q16": cost if feasible else None,
        "route": tuple(route),
        "status": int(solved.status),
    }


def _call(case, capacity, method):
    graph, start, goals, risk, deadline = case
    if method == "milp":
        return milp_route(arcs(graph, risk), start, goals, deadline)
    result = deadline_route(
        graph,
        start,
        goals,
        risk,
        available_steps=deadline,
        walking_speed_m_s=1,
        dt_s=1,
        max_labels=capacity,
        arena=MCUArena(200 * 1024),
        strategy=method,
    )
    row = asdict(result)
    row["cost_q16"] = (
        round(result.estimated_objective * 65536) if result.feasible else None
    )
    return row


def benchmark_case(case, seed, family, capacity):
    graph, start, goals, risk, deadline = case
    optimum = dp_optimum(arcs(graph, risk), start, goals, deadline)
    methods = list(METHODS)
    np.random.default_rng(seed + capacity).shuffle(methods)
    results = {}
    for method in methods:
        values = []
        row = None
        for _ in range(3):
            begun = time.perf_counter_ns()
            row = _call(case, capacity, method)
            values.append((time.perf_counter_ns() - begun) / 1e6)
        assert row is not None
        correct = row["feasible"] == (optimum is not None) and (
            not row["optimal"] or row["cost_q16"] == optimum
        )
        if row["feasible"]:
            assert row["cost_q16"] >= optimum
            if row.get("lower_bound_q16") is not None:
                assert row["lower_bound_q16"] <= optimum
        row.update(
            samples_ms=values, median_ms=float(np.median(values)), correct=correct
        )
        results[method] = row
    digest = hashlib.sha256()
    for array in (graph.edges, graph.edge_cost, risk):
        digest.update(array.tobytes())
    return {
        "seed": seed,
        "family": family,
        "label_capacity": capacity,
        "states": graph.n_nodes,
        "deadline": deadline,
        "optimum_q16": optimum,
        "input_sha256": digest.hexdigest(),
        "results": results,
    }


def summarize(rows, candidate="hybrid_exact"):
    summary = {}
    for method in METHODS:
        selected = [r for r in rows if r["optimum_q16"] is not None]
        summary[method] = {
            "correct": sum(r["results"][method]["correct"] for r in rows),
            "total": len(rows),
            "feasible_cases": len(selected),
            "proven_optimal": sum(r["results"][method]["optimal"] for r in selected),
            "mean_gap_fraction": float(
                np.mean(
                    [
                        (r["results"][method]["cost_q16"] - r["optimum_q16"])
                        / r["optimum_q16"]
                        for r in selected
                        if r["results"][method]["feasible"]
                    ]
                )
            ),
            "mean_labels": float(
                np.mean([r["results"][method].get("labels_used", 0) for r in rows])
            ),
            "median_call_ms": float(
                np.median([r["results"][method]["median_ms"] for r in rows])
            ),
        }
        pairs = [
            r
            for r in selected
            if r["results"][method]["optimal"] and r["results"][candidate]["optimal"]
        ]
        ratios = [
            r["results"][method]["median_ms"] / r["results"][candidate]["median_ms"]
            for r in pairs
        ]
        summary[method]["exact_pairs"] = len(pairs)
        if ratios:
            summary[method]["candidate_geomean_speedup_on_exact_pairs"] = float(
                np.exp(np.mean(np.log(ratios)))
            )
        if ratios:
            # Resample whole map seeds: capacities and families share inputs.
            seed_ids = sorted({r["seed"] for r in pairs})
            sums = np.array(
                [
                    sum(
                        math.log(
                            r["results"][method]["median_ms"]
                            / r["results"][candidate]["median_ms"]
                        )
                        for r in pairs
                        if r["seed"] == seed
                    )
                    for seed in seed_ids
                ]
            )
            counts = np.array(
                [sum(r["seed"] == seed for r in pairs) for seed in seed_ids]
            )
            draws = np.random.default_rng(802913).integers(
                0, len(seed_ids), size=(3000, len(seed_ids))
            )
            boot = np.exp(sums[draws].sum(axis=1) / counts[draws].sum(axis=1))
            summary[method]["speedup_seed_bootstrap_95ci"] = np.quantile(
                boot, [0.025, 0.975]
            ).tolist()
        summary[method]["candidate_lower_cost_cases"] = sum(
            r["results"][candidate]["cost_q16"] < r["results"][method]["cost_q16"]
            for r in selected
            if r["results"][candidate]["feasible"] and r["results"][method]["feasible"]
        )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    sources = (
        "ember/routing/deadline.py",
        "ember/routing/constrained.py",
        "eval/constrained_benchmark.py",
        "eval/procedural.py",
        "cascade/floorplan.py",
        "cascade/params.py",
        "cascade/rng.py",
        "cascade/extract.py",
        "eval/reference_constrained.py",
    )
    hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources}
    protocol_path = Path(__file__).with_name("constrained_protocol.json")
    if args.split == "evaluation":
        protocol = json.loads(protocol_path.read_text())
        if not protocol["frozen"] or protocol["source_files_sha256"] != hashes:
            raise RuntimeError("constrained evaluation differs from its frozen sources")
    seeds = DEVELOPMENT_SEEDS if args.split == "development" else EVALUATION_SEEDS
    # Warm imports and native HiGHS initialization equally outside case timing.
    warm = make_case(DEVELOPMENT_SEEDS[0], "tradeoff")
    for method in METHODS:
        _call(warm, 128, method)
    rows = []
    for seed in seeds:
        for family in FAMILIES:
            case = make_case(seed, family)
            for capacity in LABEL_CAPS:
                rows.append(benchmark_case(case, seed, family, capacity))
        print(f"Completed {seed}: {len(rows)} matched cases", flush=True)
    if any(
        hashlib.sha256((root / p).read_bytes()).hexdigest() != h
        for p, h in hashes.items()
    ):
        raise RuntimeError("constrained benchmark sources changed during execution")
    report = {
        "split": args.split,
        "candidate": "hybrid_exact",
        "seeds": list(seeds),
        "families": list(FAMILIES),
        "label_caps": list(LABEL_CAPS),
        "numeric_arena_bytes": 200 * 1024,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_files_sha256": hashes,
        "timing": (
            "three cold full-call samples; topology, quantization, "
            "certificates and allocation included"
        ),
        "memory_scope": (
            "bounded numeric tables and labels; Python graph/heap, "
            "SciPy matrices and HiGHS memory not MCU accounted"
        ),
        "cases": rows,
        "summary": summarize(rows),
        "by_family": {
            family: summarize([r for r in rows if r["family"] == family])
            for family in FAMILIES
        },
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        if args.split == "evaluation"
        else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
