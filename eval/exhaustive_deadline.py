"""Bounded exhaustive native checks against independent simple-path enumeration."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ember.routing.deadline import deadline_route
from eval.constrained_benchmark import _graph
from eval.paper_evidence import independently_encode
from eval.routing_certificates import verify_result
from mcu import MCUArena
from mcu.native_deadline import native_graph_search

ROOT = Path(__file__).resolve().parents[1]
MAX_DEADLINE = 6
METHODS = ("native_exact", "native_window_dp", "native_astar")


def simple_path_optima(adjacent, start, goals, maximum):
    """Enumerate every simple path; strictly positive costs make cycles useless."""
    optima = [None] * (maximum + 1)

    def visit(u, elapsed, cost, seen):
        if u in goals:
            for budget in range(elapsed, maximum + 1):
                if optima[budget] is None or cost < optima[budget]:
                    optima[budget] = cost
            return
        for v, duration, weight in adjacent[u]:
            if v not in seen and elapsed + duration <= maximum:
                visit(v, elapsed + duration, cost + weight, seen | {v})

    visit(start, 0, 0, {start})
    return optima


def no_fallback():
    raise AssertionError("exhaustive domain should fit its arena")


def check_result(model, result, optimum):
    record = asdict(result)
    record["cost_q16"] = (
        round(result.estimated_objective * 65536) if result.feasible else None
    )
    verify_result(model, record, optimum, arena_bytes=200 * 1024)
    if not result.optimal:
        raise AssertionError("unexpected cap in bounded exhaustive domain")
    return record


def run(max_nodes=4):
    checks = graphs = public_checks = 0
    fingerprint = hashlib.sha256()
    # Each unordered edge is absent or has (duration, cost) in {1,2}².
    alphabet = (None, (1, 1), (1, 2), (2, 1), (2, 2))
    for n in range(1, max_nodes + 1):
        pairs = list(itertools.combinations(range(n), 2))
        for assignments in itertools.product(alphabet, repeat=len(pairs)):
            edges, durations, weights = [], [], []
            adjacent = [[] for _ in range(n)]
            for (u, v), assignment in zip(pairs, assignments, strict=True):
                if assignment is None:
                    continue
                duration, cost = assignment
                edges.append((u, v))
                durations.append(duration)
                weights.append((cost, cost))
                adjacent[u].append([v, duration, cost])
                adjacent[v].append([u, duration, cost])
            edge_array = np.array(edges, dtype=np.int64).reshape(-1, 2)
            step_array = np.array(durations, dtype=np.int64)
            cost_array = np.array(weights, dtype=np.int64).reshape(-1, 2)
            goals = (n - 1,)
            optima = simple_path_optima(adjacent, 0, set(goals), MAX_DEADLINE)
            model = {"adjacent": adjacent, "start": 0, "goals": list(goals)}
            fingerprint.update(json.dumps(model, sort_keys=True).encode())
            for deadline, optimum in enumerate(optima):
                model["deadline"] = deadline
                for method in METHODS:
                    result = native_graph_search(
                        edge_array,
                        step_array,
                        cost_array,
                        n,
                        0,
                        goals,
                        deadline,
                        128,
                        arena=MCUArena(200 * 1024),
                        strategy=method,
                        fallback=no_fallback,
                    )
                    check_result(model, result, optimum)
                    checks += 1
            graphs += 1
        print(
            f"Exhausted through {n} nodes: {graphs} graphs, {checks} native calls",
            flush=True,
        )
    # Separate public API checks include directed arrival risk and multiple exits.
    n = max_nodes
    pairs = list(itertools.combinations(range(n), 2))
    for present in itertools.product((False, True), repeat=len(pairs)):
        edges = [edge for edge, keep in zip(pairs, present, strict=True) if keep]
        graph = _graph(n, edges, [1.0] * len(edges))
        for risks in itertools.product((0.0, 1.0), repeat=n):
            risk = np.array(risks)
            goals = tuple(sorted({min(1, n - 1), n - 1}))
            case = graph, 0, goals, risk, MAX_DEADLINE
            model = independently_encode(case)
            optima = simple_path_optima(model["adjacent"], 0, set(goals), MAX_DEADLINE)
            fingerprint.update(json.dumps(model, sort_keys=True).encode())
            for deadline, optimum in enumerate(optima):
                model["deadline"] = deadline
                result = deadline_route(
                    graph,
                    0,
                    goals,
                    risk,
                    available_steps=deadline,
                    walking_speed_m_s=1.0,
                    dt_s=1.0,
                    strategy="native_exact",
                )
                check_result(model, result, optimum)
                public_checks += 1
    files = (
        "eval/exhaustive_deadline.py",
        "eval/routing_certificates.py",
        "eval/paper_evidence.py",
        "firmware/deadline_exact.c",
        "firmware/deadline_exact.h",
        "mcu/native_deadline.py",
        "ember/routing/deadline.py",
        "eval/constrained_benchmark.py",
    )
    return {
        "schema_version": 1,
        "symmetric_graphs": graphs,
        "native_calls": checks,
        "public_api_calls": public_checks,
        "all_checks_passed": True,
        "methods": list(METHODS),
        "domain": {
            "nodes": [1, max_nodes],
            "start": 0,
            "symmetric_exit": "last node",
            "deadlines": [0, MAX_DEADLINE],
            "symmetric_edge_alphabet": [None, [1, 1], [1, 2], [2, 1], [2, 2]],
            "public_api_risk_alphabet": [0, 1],
            "public_api_edge_length": 1,
            "public_api_exits": "node 1 and last node (deduplicated)",
        },
        "models_sha256": fingerprint.hexdigest(),
        "source_sha256": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files
        },
        "scope": (
            "Finite exhaustive evidence only; not a proof for all graphs, "
            "large clocks, label exhaustion or C memory safety."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nodes", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/paper/exhaustive-checks.json"
    )
    args = parser.parse_args()
    result = run(args.max_nodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
