"""Paired development comparisons of routing and communication innovations.

Discovery and later development-generalization sets are explicit CLI inputs.
These are development experiments, never silently rebranded as sealed tests.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np

from ember.manifest import capture_manifest, write_runtime_source_bundle
from eval.harness import parse_seed_spec
from eval.procedural import (
    HELD_OUT_WORLD_SEEDS,
    ProceduralEvaluationConfig,
    build_procedural_world,
    run_policy,
    sample_exogenous_streams,
)

ALGORITHMS = {
    f"{communication}/{inference}/{route}": (communication, inference, route)
    for communication, inference, route in product(
        ("adjoint_kl", "kl_only", "occupancy_coverage"),
        ("gbp", "canonical"),
        ("lmdp", "risk_astar", "deadline_pareto", "deadline_lmdp"),
    )
}
LEGACY = "adjoint_kl/gbp/lmdp"
METRICS = (
    "rescue_completion_rate",
    "mean_unpenalized_exposure_proxy",
    "mean_dynamic_exposure_proxy",
    "mean_rescue_time_s",
    "radio_bytes",
    "routing_labels",
    "deadline_resource_fallbacks",
    "deadline_infeasible_plans",
    "deadline_interventions",
)


def evaluate_innovations(config, *, algorithms=None, root=None, progress=None):
    selected = tuple(ALGORITHMS) if algorithms is None else tuple(algorithms)
    if not selected or any(name not in ALGORITHMS for name in selected):
        raise ValueError("select at least one known algorithm")
    if len(set(selected)) != len(selected):
        raise ValueError("algorithm names must be unique")
    if HELD_OUT_WORLD_SEEDS.intersection(config.world_seeds):
        raise ValueError("innovation selection cannot use historical held-out worlds")
    before = capture_manifest(config.world_seeds[0], asdict(config), {}, root=root)
    runs = []
    for seed in config.world_seeds:
        world = build_procedural_world(seed, config)
        for noise in config.noise_seeds:
            communication_streams = {}
            methods = {}
            for name in selected:
                communication, inference, escort = ALGORITHMS[name]
                if communication not in communication_streams:
                    communication_streams[communication] = sample_exogenous_streams(
                        world,
                        noise,
                        voi_packet_budget_bytes=config.voi_packet_budget_bytes,
                        lmdp_control_temperature=config.lmdp_control_temperature,
                        communication_policy=communication,
                    )
                candidate = replace(
                    config,
                    communication_policy=communication,
                    inference_policy=inference,
                    escort_policy=escort,
                )
                methods[name] = asdict(
                    run_policy(
                        world, "ember", communication_streams[communication], candidate
                    )
                )
            reference_streams = next(iter(communication_streams.values()))
            for baseline in ("b2_flooding_dijkstra", "b3_distributed_consensus"):
                methods[baseline] = asdict(
                    run_policy(world, baseline, reference_streams, config)
                )
            runs.append(
                {
                    "world_seed": seed,
                    "noise_seed": noise,
                    "world_fingerprint": world.fingerprint,
                    "methods": methods,
                }
            )
        if progress:
            progress(seed)
    groups = defaultdict(list)
    for run in runs:
        groups[run["world_seed"]].append(run)
    indices = np.random.default_rng(config.bootstrap_seed).integers(
        0, len(groups), size=(config.bootstrap_samples, len(groups))
    )
    tail = (1.0 - config.confidence_level) / 2

    def interval(values):
        values = np.array(values, dtype=float)
        means = values[indices].mean(axis=1)
        low, high = np.quantile(means, (tail, 1 - tail))
        return {
            "mean": float(values.mean()),
            "ci_low": float(low),
            "ci_high": float(high),
        }

    names = (*selected, "b2_flooding_dijkstra", "b3_distributed_consensus")
    summary = {}
    for name in names:
        metrics = {
            metric: interval(
                [
                    np.mean([run["methods"][name][metric] for run in group])
                    for group in groups.values()
                ]
            )
            for metric in METRICS
        }
        paired = {}
        for reference in ("b2_flooding_dijkstra", "b3_distributed_consensus", LEGACY):
            if reference not in names or reference == name:
                continue
            paired[reference] = {
                metric: interval(
                    [
                        np.mean(
                            [
                                run["methods"][name][metric]
                                - run["methods"][reference][metric]
                                for run in group
                            ]
                        )
                        for group in groups.values()
                    ]
                )
                for metric in METRICS[:5]
            }
            paired[reference]["rescue_regression_scenarios"] = sum(
                run["methods"][name]["rescue_completion_rate"]
                < run["methods"][reference]["rescue_completion_rate"] - 1e-12
                for run in runs
            )
        summary[name] = {"metrics": metrics, "paired": paired}
    ranking = sorted(
        selected,
        key=lambda name: (
            -summary[name]["metrics"]["rescue_completion_rate"]["mean"],
            summary[name]["metrics"]["mean_unpenalized_exposure_proxy"]["mean"],
            summary[name]["metrics"]["mean_rescue_time_s"]["mean"],
            summary[name]["metrics"]["radio_bytes"]["mean"],
        ),
    )
    after = capture_manifest(config.world_seeds[0], asdict(config), {}, root=root)
    if before.source_identity != after.source_identity:
        raise RuntimeError(
            "source changed during experiment; rerun from a stable source"
        )
    return {
        "schema_version": 1,
        "protocol": {
            **asdict(config),
            "algorithms": list(selected),
            "phase": "development_algorithm_comparison",
            "bootstrap_unit": "world_seed",
            "paired_common_random_numbers": True,
            "source_packet_budgets_matched": True,
            "air_bytes_exactly_matched": False,
            "selection_order": [
                "rescue_completion",
                "raw_exposure",
                "rescue_time",
                "bytes",
            ],
            "future_hazard_truth_available_to_policy": False,
            "is_iso_survival": False,
            "paper_claim_ready": False,
        },
        "provenance": asdict(before),
        "runs": runs,
        "summary": summary,
        "ranking": ranking,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-seeds", default="0:7,9,19")
    parser.add_argument("--noise-seeds", default="0")
    parser.add_argument("--algorithms", help="comma-separated algorithm identifiers")
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/eval/innovation-discovery.json")
    )
    parser.add_argument("--source-bundle", type=Path)
    args = parser.parse_args(argv)
    config = ProceduralEvaluationConfig(
        world_seeds=parse_seed_spec(args.world_seeds),
        noise_seeds=parse_seed_spec(args.noise_seeds),
    )
    report = evaluate_innovations(
        config,
        algorithms=args.algorithms.split(",") if args.algorithms else None,
        progress=lambda seed: print(f"completed world {seed}", flush=True),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.source_bundle:
        digest = write_runtime_source_bundle(args.source_bundle)
        print(f"source: {args.source_bundle} (sha256 {digest})")
    for name in report["ranking"]:
        metrics = report["summary"][name]["metrics"]
        print(
            f"{name}: rescue={metrics['rescue_completion_rate']['mean']:.3f} "
            f"raw dose={metrics['mean_unpenalized_exposure_proxy']['mean']:.3f} "
            f"bytes={metrics['radio_bytes']['mean']:.0f}"
        )
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
