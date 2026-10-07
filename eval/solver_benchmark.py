"""Matched cold-start benchmarks of exact integer Bellman schedules.

Every method executes the same C row operator, with the same compiler flags,
quantized model and convergence criterion. Kernel timing includes validation,
graph preprocessing and iteration. End-to-end timing also includes quantization
and arena allocations. This evaluates solver efficiency, not rescue quality.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cascade.floorplan import build_floorplan
from cascade.params import BuildingParams
from cascade.rng import stage_rng
from ember.routing import LMDPConfig, LMDPModel, build_lmdp
from eval.procedural import _exit_nodes, _room_graph, _with_seeded_blockage
from mcu import MCUArena
from mcu.log_lmdp import LogQ16Model
from mcu.native_log_lmdp import (
    STRATEGIES,
    _library,
    _Stats,
    native_log_q16_solve,
    native_workspace_words,
    solve_native_log_q16,
)

FAMILIES = ("corridor", "grid", "sticky", "rooms", "dense", "disconnected")
DEVELOPMENT_SEEDS = tuple(range(31000, 31012))
EVALUATION_SEEDS = tuple(range(41000, 41024))


def make_model(seed: int, family: str) -> LMDPModel:
    """Seeded directed/undirected problems with randomized node numbering."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, FAMILIES.index(family)]))
    config = LMDPConfig(max_sweeps=10000)
    if family == "rooms":
        plan = build_floorplan(seed, BuildingParams())
        graph, keys = _room_graph(plan)
        exits = _exit_nodes(plan, graph, keys)
        graph, _ = _with_seeded_blockage(graph, exits, stage_rng(seed, "collapse"))
        return build_lmdp(
            graph,
            rng.uniform(0.05, 1.0, graph.n_nodes),
            exits,
            config=LMDPConfig(control_temperature=0.25, max_sweeps=10000),
        )
    if family == "dense":
        n = 64
        passive = rng.uniform(0.1, 1.0, (n, n))
        passive /= passive.sum(axis=1, keepdims=True)
        return LMDPModel(
            np.arange(n + 1) * n,
            np.tile(np.arange(n), n),
            passive.ravel(),
            rng.uniform(0.01, 0.3, n),
            np.arange(n) == n - 1,
            np.arange(n) == n - 2,
            config,
        )
    if family == "corridor":
        n = int(rng.choice([256, 512, 1024]))
        adjacency = [[(min(i + 1, n - 1), 1.0)] for i in range(n)]
        costs = rng.uniform(0.02, 2.0, n)
        exits = np.arange(n) == n - 1
        deaths = np.zeros(n, dtype=bool)
    else:
        side = int(rng.choice([12, 18, 24]))
        n = side * side
        adjacency = []
        for i in range(n):
            y, x = divmod(i, side)
            neighbours = []
            if x:
                neighbours.append(i - 1)
            if x + 1 < side:
                neighbours.append(i + 1)
            if y:
                neighbours.append(i - side)
            if y + 1 < side:
                neighbours.append(i + side)
            if family == "disconnected":
                neighbours = [j for j in neighbours if (i < n // 2) == (j < n // 2)]
            loop = 0.995 if family == "sticky" else 0.05
            weights = rng.uniform(0.5, 1.5, len(neighbours))
            weights *= (1.0 - loop) / weights.sum()
            adjacency.append([(i, loop), *zip(neighbours, weights, strict=True)])
        costs = (
            rng.uniform(0.002, 0.02, n)
            if family == "sticky"
            else rng.uniform(0.02, 0.4, n)
        )
        exits = np.arange(n) == 0
        deaths = np.arange(n) == n // 3
    # Random relabeling prevents natural Gauss-Seidel benefiting from an
    # accidentally favorable file order, equally for every method.
    permutation = rng.permutation(n)
    rows, columns, probabilities = [0], [], []
    for old in permutation:
        for target, weight in sorted(adjacency[old]):
            columns.append(int(np.flatnonzero(permutation == target)[0]))
            probabilities.append(weight)
        rows.append(len(columns))
    return LMDPModel(
        np.array(rows),
        np.array(columns),
        np.array(probabilities),
        costs[permutation],
        exits[permutation],
        deaths[permutation],
        config,
    )


def _fingerprint(model):
    digest = hashlib.sha256()
    for array in (
        model.row_ptr,
        model.col_idx,
        model.passive_prob,
        model.state_cost,
        model.exit_mask,
        model.death_mask,
    ):
        digest.update(array.tobytes())
    return digest.hexdigest()


def _measure(call, *, repeats=5, minimum_batch_s=0.008):
    started = time.perf_counter()
    call()
    elapsed = time.perf_counter() - started
    batch = min(2000, max(1, int(minimum_batch_s / max(elapsed, 1e-9))))
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        for _ in range(batch):
            call()
        samples.append((time.perf_counter_ns() - started) / batch / 1e6)
    return {
        "median_ms": float(np.median(samples)),
        "samples_ms": samples,
        "batch": batch,
    }


def benchmark_case(model, seed, family):
    storage = MCUArena(200 * 1024)
    quantized = LogQ16Model.from_float(model, storage)
    reference = solve_native_log_q16(model)
    results = {}
    methods = list(STRATEGIES)
    np.random.default_rng(seed + FAMILIES.index(family)).shuffle(methods)
    for strategy in methods:
        values = np.empty(model.n_states, np.int32)
        words = native_workspace_words(model.n_states, model.col_idx.size, strategy)
        scratch = np.empty(words, np.int32)
        stats = native_log_q16_solve(
            quantized, values, scratch, limit=model.config.max_sweeps, strategy=strategy
        )
        equal = bool(np.array_equal(values, reference.negative_log_desirability))
        # Convergence failures stay in the report; never time them as successes.
        if not reference.fixed_point or not stats.fixed_point or not equal:
            results[strategy] = {"valid": False, "stats": asdict(stats), "equal": equal}
            continue
        arrays = (
            quantized.row_ptr,
            quantized.col_idx,
            quantized.negative_log_passive,
            quantized.cost_over_temperature,
            quantized.exit_mask,
            quantized.death_mask,
        )

        def ptr(array):
            kind = ctypes.c_uint8 if array.dtype == np.uint8 else ctypes.c_int32
            return array.ctypes.data_as(ctypes.POINTER(kind))

        native_stats = _Stats()
        args = (
            model.n_states,
            model.col_idx.size,
            *(ptr(a) for a in arrays),
            model.config.max_sweeps,
            STRATEGIES[strategy],
            ptr(values),
            ptr(scratch),
            scratch.size,
            ctypes.byref(native_stats),
        )
        kernel = _library().ember_log_q16_solve

        def cold_kernel(kernel=kernel, args=args):
            status = kernel(*args)
            if status:
                raise RuntimeError(f"native benchmark error {status}")

        kernel_timing = _measure(cold_kernel)
        end_to_end = _measure(
            lambda strategy=strategy: solve_native_log_q16(model, strategy=strategy)
        )
        results[strategy] = {
            "valid": True,
            "sweeps": stats.sweeps,
            "backups": stats.backups,
            "bound_edges": stats.bound_edges,
            "selected_strategy": stats.strategy,
            "kernel": kernel_timing,
            "end_to_end": end_to_end,
            "arena_bytes": ((storage.high_water + 3) // 4) * 4
            + values.nbytes
            + scratch.nbytes,
        }
    return {
        "seed": seed,
        "family": family,
        "states": model.n_states,
        "edges": model.col_idx.size,
        "model_sha256": _fingerprint(model),
        "results": results,
    }


def summarize(cases, candidate="auto"):
    summary = {}
    for baseline in STRATEGIES:
        if baseline == candidate:
            continue
        rows = [
            row
            for row in cases
            if row["results"][candidate]["valid"] and row["results"][baseline]["valid"]
        ]
        comparison = {
            "paired_cases": len(rows),
            "total_cases": len(cases),
            "candidate_failures": sum(
                not row["results"][candidate]["valid"] for row in cases
            ),
            "baseline_failures": sum(
                not row["results"][baseline]["valid"] for row in cases
            ),
        }
        if not rows:
            summary[baseline] = comparison
            continue
        for timing in ("kernel", "end_to_end"):
            ratios = np.array(
                [
                    row["results"][baseline][timing]["median_ms"]
                    / row["results"][candidate][timing]["median_ms"]
                    for row in rows
                ]
            )
            comparison[timing] = {
                "geometric_mean_speedup": float(np.exp(np.mean(np.log(ratios)))),
                "median_speedup": float(np.median(ratios)),
                "wins": int(np.count_nonzero(ratios > 1.0)),
                "worst_speedup": float(np.min(ratios)),
                "by_family": {
                    family: float(
                        np.exp(
                            np.mean(
                                np.log(
                                    [
                                        ratios[i]
                                        for i, row in enumerate(rows)
                                        if row["family"] == family
                                    ]
                                )
                            )
                        )
                    )
                    for family in FAMILIES
                },
            }
            seed_means = np.array(
                [
                    np.mean(
                        np.log(
                            [
                                ratios[i]
                                for i, row in enumerate(rows)
                                if row["seed"] == seed
                            ]
                        )
                    )
                    for seed in sorted({row["seed"] for row in rows})
                ]
            )
            bootstrap = (
                np.random.default_rng(49381)
                .choice(seed_means, size=(2000, len(seed_means)), replace=True)
                .mean(axis=1)
            )
            comparison[timing]["seed_cluster_bootstrap_95ci"] = np.exp(
                np.quantile(bootstrap, [0.025, 0.975])
            ).tolist()
        summary[baseline] = comparison
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    seeds = DEVELOPMENT_SEEDS if args.split == "development" else EVALUATION_SEEDS
    root = Path(__file__).resolve().parents[1]
    tracked = (
        "firmware/log_lmdp.c",
        "firmware/log_lmdp.h",
        "firmware/log_lmdp_lut.h",
        "mcu/log_lmdp.py",
        "mcu/native_log_lmdp.py",
        "eval/solver_benchmark.py",
        "ember/routing/lmdp.py",
        "eval/procedural.py",
        "cascade/floorplan.py",
        "cascade/params.py",
        "cascade/rng.py",
        "cascade/extract.py",
    )
    identities = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in tracked
    }
    protocol_path = Path(__file__).with_name("solver_protocol.json")
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    if args.split == "evaluation":
        if not protocol or not protocol.get("frozen"):
            raise RuntimeError("evaluation requires a committed frozen solver protocol")
        if (
            protocol["source_files_sha256"] != identities
            or protocol["evaluation_seeds"] != list(seeds)
            or protocol["families"] != list(FAMILIES)
            or protocol["candidate"] != "auto"
        ):
            raise RuntimeError("implementation or inputs differ from frozen evaluation")
    cases = []
    for seed in seeds:
        for family in FAMILIES:
            cases.append(benchmark_case(make_model(seed, family), seed, family))
        print(f"Completed seed {seed}: {len(cases)} matched models", flush=True)
    report = {
        "split": args.split,
        "candidate": "auto",
        "seeds": list(seeds),
        "families": list(FAMILIES),
        "source_sha256": identities["firmware/log_lmdp.c"],
        "source_files_sha256": identities,
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        if protocol
        else None,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "compiler": subprocess.check_output(
            [shutil.which("cc"), "--version"], text=True
        ),
        "compiler_flags": ["-std=c99", "-O2", "-shared", "-fPIC"],
        "convergence": "exact integer fixed point, cap=10000",
        "timing": "cold solves; compilation excluded; preprocessing included",
        "cases": cases,
        "summary": summarize(cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if any(
        hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
        for name, digest in identities.items()
    ):
        raise RuntimeError("benchmark sources changed during the comparison")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
