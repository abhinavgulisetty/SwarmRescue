"""Audit frozen 0.6.0 evidence; never tune or rewrite its original benchmark."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from eval.constrained_benchmark import make_case
from eval.routing_certificates import (
    bellman_table,
    read_json,
    require,
    verify_result,
    verify_table,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independently_encode(case):
    graph, start, goals, risk, deadline = case
    adjacent = [[] for _ in range(graph.n_nodes)]
    for (a, b), length in zip(graph.edges, graph.edge_cost, strict=True):
        a, b = int(a), int(b)
        duration = max(1, math.ceil(float(length)))
        for u, v in ((a, b), (b, a)):
            cost = max(1, round((float(length) + float(risk[v]) * duration) * 65536))
            adjacent[u].append([v, duration, cost])
    return {
        "adjacent": adjacent,
        "start": int(start),
        "goals": list(goals),
        "deadline": int(deadline),
    }


def holm_adjust(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, previous = {}, 0.0
    for rank, key in enumerate(ordered):
        previous = max(previous, min(1.0, (len(ordered) - rank) * pvalues[key]))
        adjusted[key] = previous
    return adjusted


def paired_analysis(rows, method, candidate, *, comparisons=12):
    pairs = [
        r
        for r in rows
        if r["optimum_q16"] is not None
        and r["results"][method]["optimal"]
        and r["results"][candidate]["optimal"]
    ]
    if not pairs:
        return {"exact_pairs": 0}
    seeds = sorted({r["seed"] for r in pairs})
    logs = np.array(
        [
            math.log(
                r["results"][method]["median_ms"] / r["results"][candidate]["median_ms"]
            )
            for r in pairs
        ]
    )
    sums = np.array(
        [
            sum(x for x, r in zip(logs, pairs, strict=True) if r["seed"] == seed)
            for seed in seeds
        ]
    )
    counts = np.array([sum(r["seed"] == seed for r in pairs) for seed in seeds])
    # Retain all related families/capacities together, never count them as IID.
    draws = np.random.default_rng(802913).integers(0, len(seeds), (20000, len(seeds)))
    bootstrap = np.exp(sums[draws].sum(1) / counts[draws].sum(1))
    signs = np.array(list(itertools.product((-1, 1), repeat=len(seeds))))
    observed = abs(float(sums.sum()))
    pvalue = float(np.mean(np.abs(signs @ sums) >= observed - 1e-12))
    leave_one = [
        math.exp((sums.sum() - s) / (counts.sum() - c))
        for s, c in zip(sums, counts, strict=True)
        if counts.sum() > c
    ]
    return {
        "exact_pairs": len(pairs),
        "independent_seed_blocks": len(seeds),
        "candidate_geomean_speedup": math.exp(float(logs.mean())),
        "candidate_faster_pairs": int(np.sum(logs > 0)),
        "candidate_slower_pairs": int(np.sum(logs < 0)),
        "seed_bootstrap_95ci_20000_draws": np.quantile(
            bootstrap, [0.025, 0.975]
        ).tolist(),
        "bonferroni_simultaneous_95ci_20000_draws": np.quantile(
            bootstrap, [0.025 / comparisons, 1 - 0.025 / comparisons]
        ).tolist(),
        "two_sided_block_sign_flip_p": pvalue,
        "leave_one_seed_out_speedup_range": [min(leave_one), max(leave_one)],
        "equal_seed_weight_speedup": math.exp(float(np.mean(sums / counts))),
        "seed_log_sums": dict(zip(map(str, seeds), sums.tolist(), strict=True)),
        "seed_pair_counts": dict(zip(map(str, seeds), counts.tolist(), strict=True)),
    }


def audit(raw_path, protocol_path):
    raw = read_json(raw_path)
    protocol = read_json(protocol_path)
    require(raw["protocol_sha256"] == sha256(protocol_path), "protocol digest mismatch")
    require(
        raw["source_files_sha256"] == protocol["source_files_sha256"], "source manifest"
    )
    for name, digest in protocol["source_files_sha256"].items():
        require(sha256(ROOT / name) == digest, f"source changed: {name}")
    require(raw["seeds"] == protocol["evaluation_seeds"], "evaluation seeds")
    require(raw["families"] == protocol["families"], "evaluation families")
    require(raw["label_caps"] == protocol["label_caps"], "evaluation caps")
    expected_keys = set(
        itertools.product(raw["seeds"], raw["families"], raw["label_caps"])
    )
    actual_keys = [(r["seed"], r["family"], r["label_capacity"]) for r in raw["cases"]]
    require(
        len(actual_keys) == len(set(actual_keys)) and set(actual_keys) == expected_keys,
        "missing or duplicated cases",
    )
    maps = {}
    states = 0
    verified_results = 0
    for row in raw["cases"]:
        key = row["seed"], row["family"]
        if key not in maps:
            case = make_case(*key)
            graph, _, _, risk, _ = case
            digest = hashlib.sha256()
            for array in (graph.edges, graph.edge_cost, risk):
                digest.update(array.tobytes())
            model = independently_encode(case)
            values = bellman_table(model)
            optimum = verify_table(model, values)
            maps[key] = {
                "seed": key[0],
                "family": key[1],
                "model": model,
                "values": values,
                "optimum_q16": optimum,
                "input_sha256": digest.hexdigest(),
                "records": [],
            }
            states += len(values) * len(model["adjacent"])
        certificate = maps[key]
        model, optimum = certificate["model"], certificate["optimum_q16"]
        require(row["input_sha256"] == certificate["input_sha256"], "input digest")
        require(row["deadline"] == model["deadline"], "deadline changed")
        require(row["states"] == len(model["adjacent"]), "node count changed")
        require(row["optimum_q16"] == optimum, "original oracle disagrees")
        require(set(row["results"]) == set(protocol["methods"]), "method set")
        for method, result in row["results"].items():
            timings = result["samples_ms"]
            require(
                len(timings) == protocol["samples_per_case"]
                and all(
                    type(t) in (float, int) and math.isfinite(t) and t > 0
                    for t in timings
                ),
                "invalid timing samples",
            )
            require(
                result["median_ms"] == float(np.median(timings)), "incorrect median"
            )
            verify_result(
                model, result, optimum, arena_bytes=raw["numeric_arena_bytes"]
            )
            fields = (
                "route",
                "feasible",
                "optimal",
                "cost_q16",
                "travel_steps",
                "lower_bound_q16",
                "arena_high_water_bytes",
            )
            certificate["records"].append(
                {
                    "method": method,
                    "label_capacity": row["label_capacity"],
                    "result": {k: result[k] for k in fields if k in result},
                }
            )
            verified_results += 1
        if row["label_capacity"] == raw["label_caps"][-1]:
            print(f"Verified {key[0]} {key[1]}", flush=True)
    candidate = raw["candidate"]
    methods = [m for m in protocol["methods"] if m != candidate]
    statistics = {
        m: paired_analysis(raw["cases"], m, candidate, comparisons=len(methods))
        for m in methods
    }
    adjusted = holm_adjust(
        {m: x["two_sided_block_sign_flip_p"] for m, x in statistics.items()}
    )
    for method, entry in statistics.items():
        entry["holm_adjusted_p"] = adjusted[method]
    common = [
        r
        for r in raw["cases"]
        if r["optimum_q16"] is not None
        and all(r["results"][m]["optimal"] for m in protocol["methods"])
    ]
    quality = {}
    for method in protocol["methods"]:
        feasible = [r for r in raw["cases"] if r["optimum_q16"] is not None]
        returned = [r for r in feasible if r["results"][method]["feasible"]]
        gaps = [
            (r["results"][method]["cost_q16"] / r["optimum_q16"] - 1) for r in returned
        ]
        quality[method] = {
            "feasible_instances": len(feasible),
            "returned_feasible": len(returned),
            "proven_optimal": sum(r["results"][method]["optimal"] for r in returned),
            "unresolved_feasible": len(feasible) - len(returned),
            "mean_gap_fraction_returned_only": float(np.mean(gaps)),
            "worst_gap_fraction_returned_only": max(gaps),
        }
    limited = [
        {
            "seed": r["seed"],
            "family": r["family"],
            "label_capacity": r["label_capacity"],
            "upper_bound_q16": r["results"][candidate]["cost_q16"],
            "lower_bound_q16": r["results"][candidate].get("lower_bound_q16"),
            "optimum_q16": r["optimum_q16"],
            "gap_fraction": r["results"][candidate]["cost_q16"] / r["optimum_q16"] - 1,
        }
        for r in raw["cases"]
        if r["optimum_q16"] is not None and not r["results"][candidate]["optimal"]
    ]
    report = {
        "schema_version": 1,
        "candidate": candidate,
        "raw_sha256": sha256(raw_path),
        "protocol_sha256": sha256(protocol_path),
        "audit_source_sha256": {
            p: sha256(ROOT / p)
            for p in ("eval/paper_evidence.py", "eval/routing_certificates.py")
        },
        "verified_maps": len(maps),
        "verified_cases": len(raw["cases"]),
        "verified_results": verified_results,
        "verified_bellman_states": states,
        "statistics": statistics,
        "quality": quality,
        "candidate_capped_cases": limited,
        "by_family": {
            f: paired_analysis(
                [r for r in raw["cases"] if r["family"] == f], "native_astar", candidate
            )
            for f in raw["families"]
        },
        "common_support_pairs": len(common),
        "common_support_statistics": {
            m: paired_analysis(common, m, candidate, comparisons=len(methods))
            for m in methods
        },
        "limitations": [
            "Post-hoc audit of existing evidence, not a new preregistered experiment.",
            "Only final-repeat result status was preserved; "
            "other repeat statuses cannot be verified.",
            "12 map-seed clusters on one host; families and capacities are correlated.",
            "Sign flips assume independent symmetric null seed effects; "
            "randomizing timing order alone does not establish this.",
            "Holm controls multiplicity only when underlying p-values are valid; "
            "it does not remove selection or timing bias.",
            "Bootstrap intervals are approximate, particularly simultaneous "
            "tail intervals with 12 clusters.",
            "Exact-pair speed ratios condition on solver success; "
            "quality and failures must also be reported.",
            "Certificates prove the recorded discrete instances, "
            "not the complete C/Python implementation or physical model.",
        ],
    }
    certificates = {
        "schema_version": 1,
        "raw_sha256": sha256(raw_path),
        "numeric_arena_bytes": raw["numeric_arena_bytes"],
        "maps": list(maps.values()),
    }
    return report, certificates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "docs/benchmarks/native-deadline-evaluation.json.gz",
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "eval/native_deadline_protocol.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/paper/evidence-audit.json"
    )
    parser.add_argument(
        "--certificates",
        type=Path,
        default=ROOT / "docs/paper/instance-certificates.json.gz",
    )
    args = parser.parse_args(argv)
    report, certificates = audit(args.raw, args.protocol)
    write_json(args.certificates, certificates)
    report["certificates_sha256"] = sha256(args.certificates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "verified_maps",
                    "verified_cases",
                    "verified_results",
                    "verified_bellman_states",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
