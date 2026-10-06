"""Development-only range localization under actual radio loss and partitions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ember.localization import RSSIConfig, sample_rssi
from ember.manifest import capture_manifest, write_source_bundle
from ember.radio import RadioConfig, RadioMedium
from ember.range_gbp import RangeFactorGBP
from ember.range_team import RangeLocalizationTeam
from eval.harness import parse_seed_spec


def run_case(seed, loss, *, max_rounds=3):
    world_rng = np.random.default_rng(seed)
    anchors = np.array(
        [[0, 0], [12, 0], [0, 12], [12, 12], [6, 0], [0, 6], [12, 6], [6, 12]], float
    )
    anchors += world_rng.uniform(-0.5, 0.5, (8, 2))
    truths = world_rng.uniform(2, 10, (8, 2))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 713]))
    measured = anchors + rng.normal(0, 0.05, anchors.shape)
    covariance = np.tile(np.eye(2) * 0.05**2, (8, 1, 1))
    config = RSSIConfig(noise_std_db=0.45)
    readings = np.column_stack(
        [sample_rssi(anchors, truth, rng, config=config) for truth in truths]
    )
    # The full-factor control uses exactly the on-wire float32 values.
    readings = readings.astype(np.float32)
    reference = RangeFactorGBP(measured, covariance, 8)
    for source in range(8):
        for beacon in range(8):
            reference.observe_rssi(
                source, beacon, 0, float(readings[source, beacon]), config=config
            )
    control = reference.solve()
    if not control.converged:
        raise RuntimeError("complete-factor control did not converge")
    radio = RadioMedium(
        measured, loss_seed=seed + 8192, config=RadioConfig(packet_error_rate=loss)
    )
    team = RangeLocalizationTeam(measured, covariance, 8, radio, sensor_config=config)
    result = team.exchange(readings, rng, max_rounds=max_rounds)
    errors, covered, positions = [], [], []
    for solution in result.solutions:
        if solution is None or not solution.converged:
            continue
        for estimate, truth in zip(solution.estimates, truths, strict=True):
            errors.append(float(np.linalg.norm(estimate.position_m - truth)))
            covered.append(estimate.covers_95(truth))
        positions.append([estimate.position_m for estimate in solution.estimates])
    reference_positions = np.array(
        [estimate.position_m for estimate in control.estimates]
    )
    difference = (
        float(np.max(np.abs(np.array(positions) - reference_positions)))
        if positions
        else None
    )
    return {
        "world_seed": seed,
        "packet_error_rate": loss,
        "available": result.available,
        "available_receivers": sum(
            solution is not None and solution.converged for solution in result.solutions
        ),
        "rounds": result.rounds,
        "factor_counts": result.factor_counts,
        "radio_stats": asdict(result.radio_stats),
        "failures": result.failures,
        "mean_error_m": float(np.mean(errors)) if errors else None,
        "coverage_95": float(np.mean(covered)) if covered else None,
        "complete_factor_mean_error_m": float(
            np.mean(np.linalg.norm(reference_positions - truths, axis=1))
        ),
        "max_receiver_coordinate_difference_from_complete_factors_m": difference,
        "receiver_numeric_bytes": result.receiver_numeric_bytes,
        "team_numeric_bytes": result.team_numeric_bytes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-seeds", default="0:19")
    parser.add_argument("--loss-rates", default="0,0.1,0.25,0.5,1")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/eval/range-delivery.json")
    )
    args = parser.parse_args(argv)
    seeds = parse_seed_spec(args.world_seeds)
    if not seeds or any(seed not in range(200) for seed in seeds):
        parser.error("only development world seeds 0..199 are permitted")
    try:
        losses = tuple(float(x) for x in args.loss_rates.split(","))
    except ValueError:
        parser.error("loss rates must be comma-separated numbers")
    if not losses or any(
        not np.isfinite(loss) or not 0 <= loss <= 1 for loss in losses
    ):
        parser.error("loss rates must be finite in [0,1]")
    if not 1 <= args.rounds <= 8:
        parser.error("rounds must be in 1..8")
    config = {
        "world_seeds": seeds,
        "packet_error_rates": losses,
        "max_rounds": args.rounds,
        "robots": 8,
        "beacons": 8,
        "pose_std_m": 0.05,
        "rssi_std_db": 0.45,
    }
    before = capture_manifest(0, config, {})
    cases = []
    for seed in seeds:
        for loss in losses:
            cases.append(run_case(seed, loss, max_rounds=args.rounds))
        if len(cases) % 25 == 0:
            print(f"completed {len(cases)} radio cases", flush=True)
    summary = []
    for loss in losses:
        selected = [case for case in cases if case["packet_error_rate"] == loss]
        solved = [case for case in selected if case["mean_error_m"] is not None]
        summary.append(
            {
                "packet_error_rate": loss,
                "cases": len(selected),
                "available_teams": sum(case["available"] for case in selected),
                "available_receivers": sum(
                    case["available_receivers"] for case in selected
                ),
                "mean_error_m_when_solved": float(
                    np.mean([case["mean_error_m"] for case in solved])
                )
                if solved
                else None,
                "mean_coverage_95_when_solved": float(
                    np.mean([case["coverage_95"] for case in solved])
                )
                if solved
                else None,
                "mean_air_bytes": float(
                    np.mean([case["radio_stats"]["bytes_air"] for case in selected])
                ),
                "mean_rounds": float(np.mean([case["rounds"] for case in selected])),
                "max_difference_from_complete_factors_m": max(
                    (
                        case[
                            "max_receiver_coordinate_difference_from_complete_factors_m"
                        ]
                        for case in solved
                    ),
                    default=None,
                ),
            }
        )
    after = capture_manifest(0, config, {})
    if (before.git_sha, before.dirty_diff_hash) != (
        after.git_sha,
        after.dirty_diff_hash,
    ):
        raise RuntimeError("source changed during range delivery validation")
    report = {
        "schema_version": 1,
        "config": config,
        "summary": summary,
        "cases": cases,
        "provenance": asdict(after),
        "assignment_transport_modeled": False,
        "anchor_prior_transport_modeled": False,
        "whole_embedded_memory_gate_met": False,
        "whole_phase1_acceptance_gate_met": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    write_source_bundle(args.out.with_suffix(".source.tar.gz"))
    print(json.dumps(summary, indent=2))
    # Partition unavailability is an expected result, not a passing rescue.
    valid = all(
        case["available"]
        and case["max_receiver_coordinate_difference_from_complete_factors_m"] < 1e-8
        for case in cases
        if case["packet_error_rate"] == 0
    )
    valid &= all(
        case["available_receivers"] == 0
        for case in cases
        if case["packet_error_rate"] == 1
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
