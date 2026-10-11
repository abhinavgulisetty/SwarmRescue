"""Fresh cold-call comparison of compiled hybrid and traditional deadline solvers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from eval.constrained_benchmark import (
    FAMILIES,
    LABEL_CAPS,
    _call,
    arcs,
    dp_optimum,
    make_case,
)

METHODS = (
    "pareto",
    "astar",
    "lazy_astar",
    "dual_astar",
    "bounded_dp",
    "window_dp",
    "hybrid_exact",
    "native_astar",
    "native_full_dp",
    "native_window_dp",
    "native_dp",
    "native_exact",
    "milp",
)
DEVELOPMENT_SEEDS = tuple(range(72000, 72004))
EVALUATION_SEEDS = tuple(range(82000, 82012))
CANDIDATE = "native_exact"
SOURCES = (
    "firmware/deadline_exact.c",
    "firmware/deadline_exact.h",
    "mcu/native_deadline.py",
    "mcu/__init__.py",
    "ember/routing/deadline.py",
    "ember/routing/constrained.py",
    "eval/native_deadline_benchmark.py",
    "eval/constrained_benchmark.py",
    "eval/procedural.py",
    "cascade/floorplan.py",
    "cascade/params.py",
    "cascade/rng.py",
    "cascade/extract.py",
)


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in SOURCES}


def benchmark(case, seed, family, capacity):
    graph, start, goals, risk, deadline = case
    optimum = dp_optimum(arcs(graph, risk), start, goals, deadline)
    methods = list(METHODS)
    np.random.default_rng(seed + capacity).shuffle(methods)
    results = {}
    for method in methods:
        samples = []
        for _ in range(5):
            begun = time.perf_counter_ns()
            result = _call(case, capacity, method)
            samples.append((time.perf_counter_ns() - begun) / 1e6)
        correct = result["feasible"] == (optimum is not None) and (
            not result["optimal"] or result["cost_q16"] == optimum
        )
        if result["feasible"]:
            assert result["cost_q16"] >= optimum
            if result.get("lower_bound_q16") is not None:
                assert result["lower_bound_q16"] <= optimum
        result.update(
            correct=correct, samples_ms=samples, median_ms=float(np.median(samples))
        )
        results[method] = result
    digest = hashlib.sha256()
    for a in (graph.edges, graph.edge_cost, risk):
        digest.update(a.tobytes())
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


def summarize(rows):
    summary = {}
    feasible = [r for r in rows if r["optimum_q16"] is not None]
    for method in METHODS:
        pairs = [
            r
            for r in feasible
            if r["results"][method]["optimal"] and r["results"][CANDIDATE]["optimal"]
        ]
        ratios = [
            r["results"][method]["median_ms"] / r["results"][CANDIDATE]["median_ms"]
            for r in pairs
        ]
        entry = {
            "correct": sum(r["results"][method]["correct"] for r in rows),
            "total": len(rows),
            "feasible_cases": len(feasible),
            "proven_optimal": sum(r["results"][method]["optimal"] for r in feasible),
            "exact_pairs": len(pairs),
            "mean_gap_fraction": float(
                np.mean(
                    [
                        (r["results"][method]["cost_q16"] - r["optimum_q16"])
                        / r["optimum_q16"]
                        for r in feasible
                        if r["results"][method]["feasible"]
                    ]
                )
            ),
            "median_call_ms": float(
                np.median([r["results"][method]["median_ms"] for r in rows])
            ),
        }
        if ratios:
            entry["candidate_geomean_speedup"] = float(np.exp(np.mean(np.log(ratios))))
            entry["candidate_faster_pairs"] = sum(x > 1 for x in ratios)
            seeds = sorted({r["seed"] for r in pairs})
            sums = np.array(
                [
                    sum(
                        math.log(
                            r["results"][method]["median_ms"]
                            / r["results"][CANDIDATE]["median_ms"]
                        )
                        for r in pairs
                        if r["seed"] == seed
                    )
                    for seed in seeds
                ]
            )
            counts = np.array([sum(r["seed"] == seed for r in pairs) for seed in seeds])
            draws = np.random.default_rng(802913).integers(
                0, len(seeds), (3000, len(seeds))
            )
            entry["seed_bootstrap_95ci"] = np.quantile(
                np.exp(sums[draws].sum(1) / counts[draws].sum(1)), [0.025, 0.975]
            ).tolist()
        summary[method] = entry
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    hashes = source_hashes()
    protocol_path = Path(__file__).with_name("native_deadline_protocol.json")
    if args.split == "evaluation":
        protocol = json.loads(protocol_path.read_text())
        if not protocol["frozen"] or protocol["source_files_sha256"] != hashes:
            raise RuntimeError("native deadline sources differ from frozen protocol")
    warm = make_case(DEVELOPMENT_SEEDS[0], "tradeoff")
    for method in METHODS:
        _call(warm, 128, method)
    seeds = DEVELOPMENT_SEEDS if args.split == "development" else EVALUATION_SEEDS
    rows = []
    for seed in seeds:
        for family in FAMILIES:
            case = make_case(seed, family)
            for capacity in LABEL_CAPS:
                rows.append(benchmark(case, seed, family, capacity))
        print(f"Completed {seed}: {len(rows)} matched cases", flush=True)
    if source_hashes() != hashes:
        raise RuntimeError("sources changed during benchmark")
    report = {
        "split": args.split,
        "candidate": CANDIDATE,
        "seeds": list(seeds),
        "families": list(FAMILIES),
        "label_caps": list(LABEL_CAPS),
        "numeric_arena_bytes": 204800,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "source_files_sha256": hashes,
        "samples_per_case": 5,
        "timing": (
            "Full cold API call including quantization, graph assembly, "
            "allocation and preprocessing; all compiled libraries/imports "
            "warmed outside timing."
        ),
        "memory_scope": (
            "Numeric scratch only; host graph arrays, Python heap "
            "objects and native HiGHS/reference process memory excluded."
        ),
        "cases": rows,
        "summary": summarize(rows),
        "by_family": {
            f: summarize([r for r in rows if r["family"] == f]) for f in FAMILIES
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
