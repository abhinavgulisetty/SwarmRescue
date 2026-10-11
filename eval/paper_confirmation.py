"""Preregistered, all-repeat, three-process confirmation of unchanged routing."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np

from eval.constrained_benchmark import _call, make_case
from eval.native_deadline_benchmark import source_hashes
from eval.paper_evidence import independently_encode
from eval.routing_certificates import (
    bellman_table,
    read_json,
    require,
    verify_result,
    verify_table,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "eval/paper_confirmation_protocol.json"
CANDIDATE = "native_exact"
EXTRA_SOURCES = (
    "eval/paper_confirmation.py",
    "eval/paper_evidence.py",
    "eval/routing_certificates.py",
)


def hashes():
    return source_hashes() | {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in EXTRA_SOURCES
    }


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol():
    protocol = read_json(PROTOCOL)
    require(protocol["frozen"], "protocol not frozen")
    require(protocol["source_sha256"] == hashes(), "confirmation sources changed")
    return protocol


def all_exact(entry):
    return all(s["feasible"] and s["optimal"] for s in entry["samples"])


def worker(index, output):
    protocol = load_protocol()
    require(index in range(protocol["processes"]), "process index")
    methods = protocol["methods"]
    # Compile/import every measured solver before the measured fresh cases.
    development = make_case(72000, "tradeoff")
    for method in methods:
        _call(development, 128, method)
    rng = np.random.default_rng(protocol["order_seed"] + index)
    maps = list(itertools.product(protocol["seeds"], protocol["families"]))
    rng.shuffle(maps)
    certificates, rows = [], []
    for number, (seed, family) in enumerate(maps, 1):
        case = make_case(seed, family)
        model = independently_encode(case)
        values = bellman_table(model)
        optimum = verify_table(model, values)
        model_hash = hashlib.sha256(
            json.dumps(model, sort_keys=True).encode()
        ).hexdigest()
        certificates.append(
            {
                "seed": seed,
                "family": family,
                "model": model,
                "values": values,
                "optimum_q16": optimum,
                "model_sha256": model_hash,
            }
        )
        capacities = list(protocol["label_caps"])
        rng.shuffle(capacities)
        for capacity in capacities:
            results = {method: {"samples": []} for method in methods}
            order = []
            for _ in range(protocol["samples_per_process_case"]):
                round_methods = list(methods)
                rng.shuffle(round_methods)
                order.append(round_methods)
                for method in round_methods:
                    begun = time.perf_counter_ns()
                    result = _call(case, capacity, method)
                    result["elapsed_ms"] = (time.perf_counter_ns() - begun) / 1e6
                    # All verification runs outside the timed interval.
                    verify_result(
                        model,
                        result,
                        optimum,
                        arena_bytes=protocol["numeric_arena_bytes"],
                    )
                    results[method]["samples"].append(result)
            for entry in results.values():
                entry["median_ms"] = float(
                    np.median([s["elapsed_ms"] for s in entry["samples"]])
                )
            rows.append(
                {
                    "seed": seed,
                    "family": family,
                    "label_capacity": capacity,
                    "model_sha256": model_hash,
                    "optimum_q16": optimum,
                    "results": results,
                    "round_method_orders": order,
                }
            )
        if number % 20 == 0:
            print(f"Process {index}: {number}/{len(maps)} fresh maps", flush=True)
    require(hashes() == protocol["source_sha256"], "sources changed during run")
    cpu = next(
        (
            line.partition(":")[2].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "unknown",
    )
    write_json(
        output,
        {
            "schema_version": 1,
            "process_index": index,
            "pid": os.getpid(),
            "protocol_sha256": digest(PROTOCOL),
            "source_sha256": hashes(),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpu": cpu,
                "numpy": np.__version__,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "perf_counter_resolution_s": time.get_clock_info(
                    "perf_counter"
                ).resolution,
            },
            "maps": certificates,
            "cases": rows,
        },
    )
    print(f"Process {index}: wrote {output}", flush=True)


def estimate(pairs, method, *, draws=20000, comparisons=6):
    if not pairs:
        return {"exact_pairs": 0}
    seeds = sorted({key[0] for key, _ in pairs})
    logs = np.array(
        [
            [
                math.log(
                    rows[p]["results"][method]["median_ms"]
                    / rows[p]["results"][CANDIDATE]["median_ms"]
                )
                for _, rows in pairs
            ]
            for p in range(len(pairs[0][1]))
        ]
    )
    sums = np.array(
        [
            [
                sum(
                    value
                    for value, (key, _) in zip(row, pairs, strict=True)
                    if key[0] == seed
                )
                for seed in seeds
            ]
            for row in logs
        ]
    )
    counts = np.array([sum(key[0] == seed for key, _ in pairs) for seed in seeds])
    rng = np.random.default_rng(502999)
    seed_draws = rng.integers(0, len(seeds), (draws, len(seeds)))
    process_draws = rng.integers(0, len(logs), (draws, len(logs)))
    # Resample whole processes globally and whole seed blocks, preserving families/caps.
    total = sums[process_draws[:, :, None], seed_draws[:, None, :]].sum((1, 2))
    bootstrap = np.exp(total / (len(logs) * counts[seed_draws].sum(1)))
    return {
        "exact_pairs": len(pairs),
        "informative_seed_blocks": len(seeds),
        "candidate_geomean_speedup": math.exp(float(logs.mean())),
        "process_geomean_speedups": np.exp(logs.mean(1)).tolist(),
        "process_and_seed_bootstrap_95ci": np.quantile(
            bootstrap, [0.025, 0.975]
        ).tolist(),
        "bonferroni_simultaneous_95ci": np.quantile(
            bootstrap, [0.025 / comparisons, 1 - 0.025 / comparisons]
        ).tolist(),
        "candidate_faster_pairs": int(np.sum(logs.mean(0) > 0)),
        "candidate_slower_pairs": int(np.sum(logs.mean(0) < 0)),
    }


def summarize(paths, output):
    protocol = load_protocol()
    workers = sorted(
        (read_json(path) for path in paths), key=lambda w: w["process_index"]
    )
    require(
        [w["process_index"] for w in workers] == list(range(protocol["processes"])),
        "missing/duplicated process reports",
    )
    expected = set(
        itertools.product(
            protocol["seeds"], protocol["families"], protocol["label_caps"]
        )
    )
    indexes = []
    verified_samples = verified_states = 0
    for report in workers:
        require(report["protocol_sha256"] == digest(PROTOCOL), "protocol digest")
        require(report["source_sha256"] == protocol["source_sha256"], "worker sources")
        models = {}
        for certificate in report["maps"]:
            key = certificate["seed"], certificate["family"]
            require(key not in models, "duplicate map")
            model = certificate["model"]
            require(
                hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()
                == certificate["model_sha256"],
                "model hash",
            )
            optimum = verify_table(model, certificate["values"])
            require(optimum == certificate["optimum_q16"], "map optimum")
            verified_states += len(certificate["values"]) * len(model["adjacent"])
            models[key] = certificate
        index = {}
        for row in report["cases"]:
            key = row["seed"], row["family"], row["label_capacity"]
            require(key not in index, "duplicate case")
            certificate = models[key[:2]]
            require(
                row["optimum_q16"] == certificate["optimum_q16"]
                and row["model_sha256"] == certificate["model_sha256"],
                "case identity",
            )
            require(set(row["results"]) == set(protocol["methods"]), "method set")
            orders = row["round_method_orders"]
            require(
                len(orders) == protocol["samples_per_process_case"]
                and all(
                    sorted(order) == sorted(protocol["methods"]) for order in orders
                ),
                "sample orders",
            )
            for entry in row["results"].values():
                samples = entry["samples"]
                require(
                    len(samples) == protocol["samples_per_process_case"], "sample count"
                )
                times = [s["elapsed_ms"] for s in samples]
                require(all(math.isfinite(t) and t > 0 for t in times), "timing values")
                require(float(np.median(times)) == entry["median_ms"], "median")
                for result in samples:
                    verify_result(
                        certificate["model"],
                        result,
                        row["optimum_q16"],
                        arena_bytes=protocol["numeric_arena_bytes"],
                    )
                    verified_samples += 1
            index[key] = row
        require(set(index) == expected, "case coverage")
        indexes.append(index)
    combined = [(key, [index[key] for index in indexes]) for key in sorted(expected)]
    for _, rows in combined:
        require(
            len({r["model_sha256"] for r in rows}) == 1, "cross-process input mismatch"
        )
    methods = [m for m in protocol["methods"] if m != CANDIDATE]
    statistics, quality = {}, {}
    feasible = [
        (key, rows) for key, rows in combined if rows[0]["optimum_q16"] is not None
    ]
    for method in protocol["methods"]:
        quality[method] = {
            "feasible_cases": len(feasible),
            "all_repeats_feasible": sum(
                all(
                    s["feasible"]
                    for row in rows
                    for s in row["results"][method]["samples"]
                )
                for _, rows in feasible
            ),
            "all_repeats_optimal": sum(
                all(all_exact(row["results"][method]) for row in rows)
                for _, rows in feasible
            ),
            "worst_returned_gap_fraction": max(
                s["cost_q16"] / rows[0]["optimum_q16"] - 1
                for _, rows in feasible
                for row in rows
                for s in row["results"][method]["samples"]
                if s["feasible"]
            ),
        }
        if method != CANDIDATE:
            pairs = [
                (key, rows)
                for key, rows in feasible
                if all(
                    all_exact(row["results"][method])
                    and all_exact(row["results"][CANDIDATE])
                    for row in rows
                )
            ]
            statistics[method] = estimate(pairs, method, comparisons=len(methods))
    common = [
        (key, rows)
        for key, rows in feasible
        if all(
            all_exact(row["results"][method])
            for row in rows
            for method in protocol["methods"]
        )
    ]
    result = {
        "schema_version": 1,
        "protocol_sha256": digest(PROTOCOL),
        "raw_sha256": {str(path.relative_to(ROOT)): digest(path) for path in paths},
        "cases": len(combined),
        "distinct_maps": len(protocol["seeds"]) * len(protocol["families"]),
        "processes": len(workers),
        "verified_samples": verified_samples,
        "verified_bellman_states_with_process_repetition": verified_states,
        "statistics": statistics,
        "quality": quality,
        "primary_endpoint": {
            "comparator": protocol["primary_comparator"],
            "confirmed_by_preregistered_ci_rule": statistics[
                protocol["primary_comparator"]
            ]["process_and_seed_bootstrap_95ci"][0]
            > 1,
        },
        "common_support_pairs": len(common),
        "common_support_statistics": {
            m: estimate(common, m, comparisons=len(methods)) for m in methods
        },
        "by_family_vs_primary": {
            f: estimate(
                [
                    (key, rows)
                    for key, rows in feasible
                    if key[1] == f
                    and all(
                        all_exact(row["results"][CANDIDATE])
                        and all_exact(row["results"][protocol["primary_comparator"]])
                        for row in rows
                    )
                ],
                protocol["primary_comparator"],
            )
            for f in protocol["families"]
        },
        "limitations": [
            "One physical host and only three process sessions.",
            "Bootstrap intervals are approximate; "
            "no universal or worst-case speed theorem.",
            "Exact-pair timing remains conditional on success; "
            "report quality and common support.",
            "No new physical rescue experiment or measured "
            "embedded deadline-kernel cycles.",
            "No per-component ablation; "
            "component attribution and novelty are not established.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["primary_endpoint"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, choices=(0, 1, 2))
    parser.add_argument("--summarize", nargs=3, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.worker is not None:
        require(args.summarize is None, "choose worker or summarize")
        worker(args.worker, args.output)
    else:
        require(args.summarize is not None, "choose worker or summarize")
        summarize(args.summarize, args.output)


if __name__ == "__main__":
    main()
