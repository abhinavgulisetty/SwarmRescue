"""Development resilience grid for EMBER contribution C4.

The grid models robot loss and radio partition without invoking a failure detector.
Unrefreshed local factors and edge messages lose precision during an outage; a separate
partition copy restores links and measures recovery half-life.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ember.gbp import GaussianBeliefPropagation, GBPConfig
from ember.manifest import capture_manifest
from eval.harness import parse_seed_spec
from eval.procedural import (
    DEVELOPMENT_WORLD_SEEDS,
    HELD_OUT_WORLD_SEEDS,
    ProceduralEvaluationConfig,
    ProceduralWorld,
    build_procedural_world,
)


@dataclass(frozen=True, slots=True)
class ResilienceConfig:
    world_seeds: tuple[int, ...] = tuple(range(20))
    failure_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    robot_counts: tuple[int, ...] = (8, 16, 24)
    outage_sweeps: int = 12
    process_variance_per_sweep: float = 0.25
    max_recovery_sweeps: int = 500
    recovery_tolerance: float = 1.0e-5
    allow_held_out: bool = False

    def __post_init__(self) -> None:
        if not self.world_seeds:
            raise ValueError("world_seeds cannot be empty")
        if (
            not self.robot_counts
            or tuple(sorted(set(self.robot_counts))) != self.robot_counts
            or any(count < 4 for count in self.robot_counts)
        ):
            raise ValueError("robot_counts must be unique, sorted, and at least four")

        if (
            not self.failure_levels
            or tuple(sorted(set(self.failure_levels))) != self.failure_levels
            or any(not 0.0 <= level <= 1.0 for level in self.failure_levels)
        ):
            raise ValueError("failure_levels must be unique, sorted, and in [0, 1]")
        if self.outage_sweeps < 1 or self.max_recovery_sweeps < 1:
            raise ValueError("sweep counts must be positive")
        if (
            self.process_variance_per_sweep <= 0.0
            or not math.isfinite(self.process_variance_per_sweep)
            or self.recovery_tolerance <= 0.0
            or not math.isfinite(self.recovery_tolerance)
        ):
            raise ValueError("variance and tolerance must be finite and positive")
        sealed = HELD_OUT_WORLD_SEEDS.intersection(self.world_seeds)
        if sealed and not self.allow_held_out:
            raise ValueError("held-out seeds are sealed until thresholds are frozen")

    @property
    def split(self) -> str:
        selected = set(self.world_seeds)
        if selected <= DEVELOPMENT_WORLD_SEEDS:
            return "development"
        if selected <= HELD_OUT_WORLD_SEEDS:
            return "held_out"
        return "custom"


@dataclass(frozen=True, slots=True)
class ResilienceRun:
    world_seed: int
    failure_level: float
    robot_count: int
    failed_robots: int
    affected_cells: int
    affected_messages: int
    outage_sweeps: int
    partition_initial_error: float
    recovery_half_life_sweeps: int
    recovery_sweeps: int
    recovery_final_error: float
    recovered: bool
    death_variance_ratio: float
    death_risk_margin_growth: float
    failure_detector_used: bool


@dataclass(frozen=True, slots=True)
class ResilienceReport:
    schema_version: int
    protocol: dict[str, Any]
    provenance: dict[str, Any]
    runs: tuple[ResilienceRun, ...]
    summary: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def _engine(world: ProceduralWorld) -> GaussianBeliefPropagation:
    engine = GaussianBeliefPropagation(
        world.graph,
        mean=0.30,
        variance=0.04,
        config=GBPConfig(
            edge_variance=0.8,
            damping=0.6,
            tolerance=1.0e-8,
            max_sweeps=2_000,
        ),
    )
    robots = np.asarray(world.robot_nodes, dtype=np.int64)
    truth = world.dynamic_cost[world.rescue_start_step]
    engine.observe(
        robots,
        truth[robots],
        np.full(robots.size, 0.01, dtype=np.float64),
    )
    result = engine.run()
    if not result.converged:
        raise RuntimeError("reference GBP did not converge")
    return engine


def _belief_error(left, right) -> float:
    mean_error = float(np.sqrt(np.mean(np.square(left.mean - right.mean))))
    deviation_error = float(
        np.sqrt(
            np.mean(
                np.square(np.sqrt(left.variance) - np.sqrt(right.variance))
            )
        )
    )
    return mean_error + deviation_error


def _ownership(world: ProceduralWorld) -> np.ndarray:
    positions = world.graph.world_positions()
    robots = positions[np.asarray(world.robot_nodes, dtype=np.int64)]
    distance = np.linalg.norm(positions[:, None, :] - robots[None, :, :], axis=2)
    return np.argmin(distance, axis=1)


def _failure_order(world: ProceduralWorld) -> np.ndarray:
    rng = np.random.default_rng(
        np.random.SeedSequence((world.world_seed, 0xC4, len(world.robot_nodes)))
    )
    return rng.permutation(len(world.robot_nodes))


def evaluate_resilience_world(
    world: ProceduralWorld,
    config: ResilienceConfig,
) -> tuple[ResilienceRun, ...]:
    reference_engine = _engine(world)
    reference = reference_engine.belief()
    owner = _ownership(world)
    order = _failure_order(world)
    runs = []

    for level in config.failure_levels:
        failed_count = min(
            len(world.robot_nodes),
            round(level * len(world.robot_nodes)),
        )
        failed = order[:failed_count]
        affected_cells = np.isin(owner, failed)
        template = _engine(world)
        affected_messages = affected_cells[template.source] | affected_cells[
            template.target
        ]
        active_messages = ~affected_messages

        partitioned = _engine(world)
        for _ in range(config.outage_sweeps):
            partitioned.age_messages(
                affected_messages,
                config.process_variance_per_sweep,
            )
            partitioned.sweep(active_messages)
        stale = partitioned.belief()
        initial_error = _belief_error(stale, reference)
        half_threshold = initial_error / 2.0
        half_life = 0 if initial_error <= config.recovery_tolerance else -1
        recovery_sweeps = 0
        final_error = initial_error
        for sweep in range(1, config.max_recovery_sweeps + 1):
            partitioned.sweep()
            recovery_sweeps = sweep
            final_error = _belief_error(partitioned.belief(), reference)
            if half_life < 0 and final_error <= half_threshold:
                half_life = sweep
            if final_error <= config.recovery_tolerance:
                break

        dead = _engine(world)
        for _ in range(config.outage_sweeps):
            dead.age(
                np.flatnonzero(affected_cells),
                config.process_variance_per_sweep,
            )
            dead.age_messages(
                affected_messages,
                config.process_variance_per_sweep,
            )
            dead.sweep(active_messages)
        dead_belief = dead.belief()
        if np.any(affected_cells):
            variance_ratio = float(
                np.mean(
                    dead_belief.variance[affected_cells]
                    / reference.variance[affected_cells]
                )
            )
            risk_growth = float(
                np.mean(
                    np.sqrt(dead_belief.variance[affected_cells])
                    - np.sqrt(reference.variance[affected_cells])
                )
            )
        else:
            variance_ratio = 1.0
            risk_growth = 0.0

        runs.append(
            ResilienceRun(
                world_seed=world.world_seed,
                failure_level=level,
                robot_count=len(world.robot_nodes),
                failed_robots=failed_count,
                affected_cells=int(np.count_nonzero(affected_cells)),
                affected_messages=int(np.count_nonzero(affected_messages)),
                outage_sweeps=config.outage_sweeps,
                partition_initial_error=initial_error,
                recovery_half_life_sweeps=(
                    half_life if half_life >= 0 else recovery_sweeps
                ),
                recovery_sweeps=recovery_sweeps,
                recovery_final_error=final_error,
                recovered=final_error <= config.recovery_tolerance,
                death_variance_ratio=variance_ratio,
                death_risk_margin_growth=risk_growth,
                failure_detector_used=False,
            )
        )
    return tuple(runs)


def _summary(
    runs: tuple[ResilienceRun, ...],
    config: ResilienceConfig,
) -> dict[str, Any]:
    by_level = {}
    for level in config.failure_levels:
        selected = tuple(run for run in runs if run.failure_level == level)

        def mean(
            name: str, items: tuple[ResilienceRun, ...] = selected
        ) -> float:
            return float(np.mean([getattr(run, name) for run in items]))

        by_level[str(level)] = {
            "run_count": len(selected),
            "mean_failed_robots": mean("failed_robots"),
            "mean_recovery_half_life_sweeps": mean(
                "recovery_half_life_sweeps"
            ),
            "maximum_recovery_half_life_sweeps": int(
                max(run.recovery_half_life_sweeps for run in selected)
            ),
            "mean_recovery_final_error": mean("recovery_final_error"),
            "recovery_rate": mean("recovered"),
            "mean_death_variance_ratio": mean("death_variance_ratio"),
            "mean_death_risk_margin_growth": mean(
                "death_risk_margin_growth"
            ),
        }
    grid_complete = (
        len(config.world_seeds) >= 20
        and len(config.failure_levels) >= 5
        and max(config.robot_counts) >= 24
    )
    return {
        "world_count": len(config.world_seeds),
        "failure_level_count": len(config.failure_levels),
        "robot_count_count": len(config.robot_counts),
        "robot_counts": list(config.robot_counts),
        "scenario_count": len(runs),
        "by_failure_level": by_level,
        "claim_checks": {
            "twenty_world_five_level_grid_complete": grid_complete,
            "scale_to_24_robots_complete": max(config.robot_counts) >= 24,
            "all_partitions_recovered": all(run.recovered for run in runs),
            "uncertainty_grows_after_death": all(
                run.death_variance_ratio > 1.0
                for run in runs
                if run.failure_level > 0.0
            ),
            "failure_detector_used": False,
            "mission_integration_complete": False,
            "paper_claim_ready": False,
        },
    }


def evaluate_resilience(
    config: ResilienceConfig,
    *,
    root: Path | None = None,
) -> ResilienceReport:
    world_config = ProceduralEvaluationConfig(
        world_seeds=config.world_seeds,
        noise_seeds=(0,),
        robot_count=max(config.robot_counts),
    )
    worlds = (
        build_procedural_world(
            seed,
            world_config,
        )
        for seed in config.world_seeds
    )
    runs = tuple(
        run
        for world in worlds
        for robot_count in config.robot_counts
        for run in evaluate_resilience_world(
            replace(
                world,
                robot_nodes=world.robot_nodes[:robot_count],
            ),
            config,
        )
    )
    protocol = {
        "split": config.split,
        "world_seeds": list(config.world_seeds),
        "failure_levels": list(config.failure_levels),
        "robot_counts": list(config.robot_counts),
        "outage_sweeps": config.outage_sweeps,
        "process_variance_per_sweep": config.process_variance_per_sweep,
        "failure_detector": "none",
        "partition_model": "owner_region_message_suppression",
        "death_model": "owner_region_local_and_message_precision_aging",
        "recovery_metric": "belief_error_half_life_in_gbp_sweeps",
        "component_benchmark_only": True,
        "held_out_seeds_opened": config.split == "held_out",
    }
    manifest = capture_manifest(
        config.world_seeds[0],
        protocol,
        {"scenario_count": len(runs)},
        root=root,
    )
    provenance = {
        "created_at_utc": manifest.created_at_utc,
        "git_sha": manifest.git_sha,
        "git_dirty": manifest.git_dirty,
        "dirty_diff_hash": manifest.dirty_diff_hash,
        "lockfile_hash": manifest.lockfile_hash,
        "config_hash": manifest.config_hash,
        "versions": manifest.versions,
    }
    return ResilienceReport(
        schema_version=2,
        protocol=protocol,
        provenance=provenance,
        runs=runs,
        summary=_summary(runs, config),
    )


def write_report(report: ResilienceReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report.to_json(), encoding="utf-8")
    temporary.replace(path)


def write_csv(report: ResilienceReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(asdict(report.runs[0]).keys()),
        )
        writer.writeheader()
        for run in report.runs:
            writer.writerow(asdict(run))
    temporary.replace(path)


def format_summary(report: ResilienceReport) -> str:
    checks = report.summary["claim_checks"]
    highest = report.summary["by_failure_level"]["1.0"]
    return "\n".join(
        (
            "C4 resilience scenarios: "
            f"{report.summary['scenario_count']} "
            f"({report.summary['world_count']} worlds x "
            f"{report.summary['failure_level_count']} levels x "
            f"{report.summary['robot_count_count']} robot counts)",
            "100% partition mean half-life: "
            f"{highest['mean_recovery_half_life_sweeps']:.2f} sweeps",
            "100% death variance ratio: "
            f"{highest['mean_death_variance_ratio']:.3f}x",
            "all partitions recovered: "
            f"{checks['all_partitions_recovered']}",
            "paper claim: NOT READY",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-seeds", default="0:19")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/eval/phase2-resilience.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/eval/phase2-resilience.csv"),
    )
    parser.add_argument("--allow-held-out", action="store_true")
    args = parser.parse_args(argv)
    report = evaluate_resilience(
        ResilienceConfig(
            world_seeds=parse_seed_spec(args.world_seeds),
            allow_held_out=args.allow_held_out,
        )
    )
    write_report(report, args.out)
    write_csv(report, args.csv)
    print(format_summary(report))
    print(f"report: {args.out}")
    print(f"runs:   {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
