"""Development-only paired localization benchmark on shared sensor draws."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ember.localization import RSSIConfig, localize_rssi, sample_rssi
from ember.manifest import capture_manifest, write_source_bundle
from ember.range_gbp import RangeFactorGBP
from eval.harness import parse_seed_spec


def run_case(world_seed, noise_seed, *, pose_std_m=0.05):
    world_rng = np.random.default_rng(world_seed)
    truth_anchors = np.array(
        [
            [0, 0],
            [12, 0],
            [0, 12],
            [12, 12],
            [6, 0],
            [0, 6],
            [12, 6],
            [6, 12],
        ],
        dtype=float,
    ) + world_rng.uniform(-0.5, 0.5, (8, 2))
    truth_beacons = world_rng.uniform(2, 10, (8, 2))
    noise_rng = np.random.default_rng(
        np.random.SeedSequence([world_seed, noise_seed, 913])
    )
    anchors = truth_anchors + noise_rng.normal(0, pose_std_m, (8, 2))
    cfg = RSSIConfig(noise_std_db=(0.15, 0.45, 1.5)[noise_seed % 3])
    covariance = np.tile(np.eye(2) * max(pose_std_m**2, 1e-6), (8, 1, 1))
    engine = RangeFactorGBP(anchors, covariance, 8)
    classical = []
    for beacon, truth in enumerate(truth_beacons):
        rssi = sample_rssi(truth_anchors, truth, noise_rng, config=cfg)
        classical.append(localize_rssi(anchors, rssi, config=cfg))
        for source, reading in enumerate(rssi):
            engine.observe_rssi(source, beacon, 0, float(reading), config=cfg)
    solution = engine.solve()

    def score(estimates):
        errors = [
            float(np.linalg.norm(estimate.position_m - truth))
            for estimate, truth in zip(estimates, truth_beacons, strict=True)
        ]
        covered = [
            estimate.covers_95(truth)
            for estimate, truth in zip(estimates, truth_beacons, strict=True)
        ]
        return {
            "mean_error_m": float(np.mean(errors)),
            "max_error_m": max(errors),
            "coverage_95": float(np.mean(covered)),
        }

    return {
        "world_seed": world_seed,
        "noise_seed": noise_seed,
        "noise_std_db": cfg.noise_std_db,
        "classical": score(classical),
        "range_gbp": score(solution.estimates),
        "converged": solution.converged,
        "sweeps": solution.sweeps,
        "linearizations": solution.linearizations,
        "retained_numeric_bytes": solution.retained_numeric_bytes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-seeds", default="0:199")
    parser.add_argument("--noise-seeds", default="0:2")
    parser.add_argument("--pose-std-m", type=float, default=0.05)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/eval/localization-gbp.json")
    )
    args = parser.parse_args(argv)
    worlds, noises = (
        parse_seed_spec(args.world_seeds),
        parse_seed_spec(args.noise_seeds),
    )
    if not worlds or any(seed not in range(200) for seed in worlds):
        parser.error("only development world seeds 0..199 are permitted")
    if not noises or any(seed < 0 for seed in noises):
        parser.error("noise seeds must be nonnegative")
    if not np.isfinite(args.pose_std_m) or args.pose_std_m < 0:
        parser.error("pose standard deviation must be finite and nonnegative")
    config = {
        "world_seeds": worlds,
        "noise_seeds": noises,
        "pose_std_m": args.pose_std_m,
    }
    before = capture_manifest(0, config, {})
    cases = []
    for world in worlds:
        for noise in noises:
            cases.append(run_case(world, noise, pose_std_m=args.pose_std_m))
        if len(cases) % 30 == 0:
            print(f"completed {len(cases)} paired cases", flush=True)
    differences = np.array(
        [
            np.mean(
                [
                    case["range_gbp"]["mean_error_m"]
                    - case["classical"]["mean_error_m"]
                    for case in cases
                    if case["world_seed"] == world
                ]
            )
            for world in worlds
        ]
    )
    bootstrap_rng = np.random.default_rng(8192)
    bootstrap = np.mean(
        bootstrap_rng.choice(differences, (2000, len(differences))), axis=1
    )
    summary = {
        "paired_cases": len(cases),
        "beacons_per_case": 8,
        "range_gbp_converged": sum(case["converged"] for case in cases),
        "paired_world_mean_error_difference_m": float(np.mean(differences)),
        "paired_world_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "max_retained_numeric_bytes": max(
            case["retained_numeric_bytes"] for case in cases
        ),
    }
    for method in ("classical", "range_gbp"):
        summary[method] = {
            metric: float(np.mean([case[method][metric] for case in cases]))
            for metric in ("mean_error_m", "max_error_m", "coverage_95")
        }
    after = capture_manifest(0, config, {})
    if (before.git_sha, before.dirty_diff_hash) != (
        after.git_sha,
        after.dirty_diff_hash,
    ):
        raise RuntimeError("source changed during localization evaluation")
    report = {
        "schema_version": 1,
        "config": config,
        "summary": summary,
        "cases": cases,
        "provenance": asdict(after),
        "distributed_localization_transport": False,
        "whole_embedded_memory_gate_met": False,
        "sensor_calibration_gate_met": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    write_source_bundle(args.out.with_suffix(".source.tar.gz"))
    print(json.dumps(summary, indent=2))
    return 0 if summary["range_gbp_converged"] == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
