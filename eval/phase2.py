"""Development-only component benchmark for EMBER Phase 2.

This benchmark deliberately evaluates C2 and C3 as components before they are wired
into the complete physical mission. It never opens held-out seeds and never promotes
component results to a survival claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ember.comms import packetize_gaussians
from ember.manifest import capture_manifest
from ember.routing import (
    LMDPConfig,
    build_lmdp,
    solve_jacobi,
    solve_safeguarded_nca,
    trajectory_optimism,
)
from ember.voi import TokenBucket, schedule_gaussian_updates, solve_adjoint
from eval.harness import parse_seed_spec
from eval.procedural import (
    DEVELOPMENT_WORLD_SEEDS,
    HELD_OUT_WORLD_SEEDS,
    ProceduralEvaluationConfig,
    build_procedural_world,
)


@dataclass(frozen=True, slots=True)
class Phase2BenchmarkConfig:
    world_seeds: tuple[int, ...] = tuple(range(20))
    packet_budget_bytes: int = 244
    tolerance: float = 1.0e-8
    max_sweeps: int = 10_000
    allow_held_out: bool = False

    def __post_init__(self) -> None:
        if not self.world_seeds:
            raise ValueError("world_seeds cannot be empty")
        if self.packet_budget_bytes < 20:
            raise ValueError("packet budget must fit one Gaussian update")
        if self.tolerance <= 0.0 or self.max_sweeps < 1:
            raise ValueError("solver tolerance and sweep cap must be positive")
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
class Phase2WorldResult:
    world_seed: int
    node_count: int
    jacobi_sweeps: int
    nca_sweeps: int
    nca_accepted: int
    nca_rejected: int
    nca_parameter_bytes: int
    nca_max_error: float
    nca_converged: bool
    residual_ratio_bound: float
    optimism_mean_gap: float
    optimism_maximum_gap: float
    adjoint_sweeps: int
    adjoint_converged: bool
    flooding_bytes: int
    voi_bytes: int
    selected_updates: int
    flooding_updates: int
    byte_reduction_ratio: float
    receiver_information_nats: float
    nats_per_byte: float


@dataclass(frozen=True, slots=True)
class Phase2BenchmarkReport:
    schema_version: int
    protocol: dict[str, Any]
    provenance: dict[str, Any]
    runs: tuple[Phase2WorldResult, ...]
    summary: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def _summary(runs: tuple[Phase2WorldResult, ...]) -> dict[str, Any]:
    def values(name: str) -> np.ndarray:
        return np.asarray([getattr(run, name) for run in runs], dtype=np.float64)

    jacobi = values("jacobi_sweeps")
    nca = values("nca_sweeps")
    flooding = values("flooding_bytes")
    voi = values("voi_bytes")
    speedup = jacobi / nca
    return {
        "world_count": len(runs),
        "nca": {
            "all_converged": all(run.nca_converged for run in runs),
            "parameter_bytes": int(runs[0].nca_parameter_bytes),
            "mean_jacobi_sweeps": float(np.mean(jacobi)),
            "mean_nca_sweeps": float(np.mean(nca)),
            "mean_sweep_speedup": float(np.mean(speedup)),
            "minimum_sweep_speedup": float(np.min(speedup)),
            "maximum_solution_error": float(
                max(run.nca_max_error for run in runs)
            ),
            "claim_threshold_met": bool(np.mean(speedup) >= 2.0),
        },
        "lambda_diagnostic": {
            "mean_optimism_gap": float(np.mean(values("optimism_mean_gap"))),
            "maximum_optimism_gap": float(np.max(values("optimism_maximum_gap"))),
            "implemented": True,
        },
        "voi": {
            "all_adjoint_converged": all(
                run.adjoint_converged for run in runs
            ),
            "mean_flooding_bytes": float(np.mean(flooding)),
            "mean_voi_bytes": float(np.mean(voi)),
            "mean_byte_reduction_ratio": float(
                np.mean(values("byte_reduction_ratio"))
            ),
            "mean_nats_per_byte": float(
                np.mean(values("nats_per_byte"))
            ),
            "method_specific_policy_implemented": True,
        },
        "claim_checks": {
            "c2_safety_gate": all(run.nca_converged for run in runs),
            "c2_speed_gate": bool(np.mean(speedup) >= 2.0),
            "c3_component_gate": all(
                run.adjoint_converged and run.voi_bytes <= run.flooding_bytes
                for run in runs
            ),
            "mission_integration_complete": False,
            "paper_claim_ready": False,
        },
    }


def evaluate_phase2(
    config: Phase2BenchmarkConfig,
    *,
    root: Path | None = None,
) -> Phase2BenchmarkReport:
    """Benchmark safe NCA and VoI scheduling on procedural room graphs."""

    runs: list[Phase2WorldResult] = []
    world_config = ProceduralEvaluationConfig(
        world_seeds=config.world_seeds,
        noise_seeds=(0,),
    )
    solver_config = LMDPConfig(
        tolerance=config.tolerance,
        max_sweeps=config.max_sweeps,
    )
    for seed in config.world_seeds:
        world = build_procedural_world(seed, world_config)
        cost = np.asarray(
            world.dynamic_cost[world.rescue_start_step],
            dtype=np.float64,
        )
        model = build_lmdp(
            world.graph,
            cost,
            world.exit_nodes,
            config=solver_config,
        )
        classical = solve_jacobi(model)
        accelerated = solve_safeguarded_nca(model)

        source = np.zeros(model.n_states, dtype=np.float64)
        optimism = trajectory_optimism(model, classical)
        source[np.asarray(world.survivor_nodes, dtype=np.int64)] = 1.0
        adjoint = solve_adjoint(model, source)

        scale = max(float(np.max(cost)), 1.0e-12)
        prior_mean = np.zeros(model.n_states)
        prior_variance = np.full(model.n_states, 4.0)
        posterior_mean = cost / scale
        posterior_variance = np.ones(model.n_states)
        cells = np.arange(model.n_states, dtype=np.int64)
        bucket = TokenBucket(
            capacity_bytes=config.packet_budget_bytes,
            refill_bytes=config.packet_budget_bytes,
        )
        schedule = schedule_gaussian_updates(
            sender=0,
            sequence=0,
            cells=cells,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mean=posterior_mean,
            posterior_variance=posterior_variance,
            adjoint=adjoint.field,
            bucket=bucket,
        )
        flooding_packets = packetize_gaussians(
            0,
            0,
            cells,
            posterior_mean,
            posterior_variance,
        )
        flooding_bytes = sum(packet.size_bytes for packet in flooding_packets)
        voi_bytes = schedule.bytes_scheduled
        runs.append(
            Phase2WorldResult(
                world_seed=seed,
                node_count=model.n_states,
                jacobi_sweeps=classical.sweeps,
                nca_sweeps=accelerated.solution.sweeps,
                nca_accepted=accelerated.accepted_proposals,
                nca_rejected=accelerated.rejected_proposals,
                nca_parameter_bytes=accelerated.parameter_bytes,
                nca_max_error=float(
                    np.max(
                        np.abs(
                            classical.desirability
                            - accelerated.solution.desirability
                        )
                    )
                ),
                nca_converged=accelerated.solution.converged,
                residual_ratio_bound=accelerated.worst_residual_ratio,
                adjoint_sweeps=adjoint.sweeps,
                adjoint_converged=adjoint.converged,
                optimism_mean_gap=optimism.mean_gap,
                optimism_maximum_gap=optimism.maximum_gap,
                flooding_bytes=flooding_bytes,
                voi_bytes=voi_bytes,
                selected_updates=len(schedule.selected_cells),
                flooding_updates=model.n_states,
                byte_reduction_ratio=(
                    1.0 - voi_bytes / flooding_bytes
                    if flooding_bytes
                    else 0.0
                ),
                receiver_information_nats=schedule.information_gain_nats,
                nats_per_byte=schedule.nats_per_byte,
            )
        )

    protocol = {
        "split": config.split,
        "world_seeds": list(config.world_seeds),
        "held_out_seeds_opened": config.split == "held_out",
        "component_benchmark_only": True,
        "packet_budget_bytes": config.packet_budget_bytes,
        "nca_safeguard": "global_bellman_residual_reference",
        "nca_training": "default_hand_initialized_int8_table",
        "voi_priority": "adjoint_times_scalar_gaussian_kl",
        "mission_integration_complete": False,
    }
    manifest = capture_manifest(
        config.world_seeds[0],
        protocol,
        {"world_count": len(runs)},
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
    run_tuple = tuple(runs)
    return Phase2BenchmarkReport(
        schema_version=2,
        protocol=protocol,
        provenance=provenance,
        runs=run_tuple,
        summary=_summary(run_tuple),
    )


def write_report(report: Phase2BenchmarkReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report.to_json(), encoding="utf-8")
    temporary.replace(path)


def write_csv(report: Phase2BenchmarkReport, path: Path) -> None:
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


def format_summary(report: Phase2BenchmarkReport) -> str:
    nca = report.summary["nca"]
    voi = report.summary["voi"]
    checks = report.summary["claim_checks"]
    return "\n".join(
        (
            f"Phase-2 component worlds: {report.summary['world_count']}",
            (
                "NCA sweeps: "
                f"{nca['mean_jacobi_sweeps']:.2f} -> "
                f"{nca['mean_nca_sweeps']:.2f} "
                f"({nca['mean_sweep_speedup']:.3f}x)"
            ),
            (
                "VoI bytes: "
                f"{voi['mean_flooding_bytes']:.1f} -> "
                f"{voi['mean_voi_bytes']:.1f} "
                f"({voi['mean_byte_reduction_ratio']:.1%} reduction)"
            ),
            f"C2 speed gate: {'PASS' if checks['c2_speed_gate'] else 'NOT READY'}",
            f"paper claim: {'READY' if checks['paper_claim_ready'] else 'NOT READY'}",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-seeds", default="0:19")
    parser.add_argument("--packet-budget", type=int, default=244)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/eval/phase2-components.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/eval/phase2-components.csv"),
    )
    parser.add_argument("--allow-held-out", action="store_true")
    args = parser.parse_args(argv)
    report = evaluate_phase2(
        Phase2BenchmarkConfig(
            world_seeds=parse_seed_spec(args.world_seeds),
            packet_budget_bytes=args.packet_budget,
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
