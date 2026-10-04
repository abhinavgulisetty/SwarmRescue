"""Development-only control-temperature selection for the LMDP.

Candidates are selected by static clairvoyant route regret on procedural worlds, with
trajectory-level optimism as the tie-breaker. Held-out worlds remain sealed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from ember.manifest import capture_manifest
from ember.routing import (
    LMDPConfig,
    build_lmdp,
    greedy_route,
    solve_jacobi,
    trajectory_optimism,
)
from eval.harness import parse_seed_spec
from eval.procedural import (
    DEVELOPMENT_WORLD_SEEDS,
    HELD_OUT_WORLD_SEEDS,
    ProceduralEvaluationConfig,
    _route_to_any,
    build_procedural_world,
)


@dataclass(frozen=True, slots=True)
class LambdaSelectionConfig:
    world_seeds: tuple[int, ...] = tuple(range(20))
    candidates: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
    allow_held_out: bool = False

    def __post_init__(self) -> None:
        if not self.world_seeds:
            raise ValueError("world_seeds cannot be empty")
        if (
            not self.candidates
            or tuple(sorted(set(self.candidates))) != self.candidates
            or any(
                value <= 0.0 or not math.isfinite(value)
                for value in self.candidates
            )
        ):
            raise ValueError("lambda candidates must be unique sorted positive values")
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
class LambdaRun:
    world_seed: int
    control_temperature: float
    survivor_count: int
    mean_static_regret: float
    maximum_static_regret: float
    mean_optimism_gap: float
    maximum_optimism_gap: float
    mean_solver_sweeps: float
    all_converged: bool


@dataclass(frozen=True, slots=True)
class LambdaSelectionReport:
    schema_version: int
    protocol: dict[str, Any]
    provenance: dict[str, Any]
    runs: tuple[LambdaRun, ...]
    summary: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def _route_objective(
    route: tuple[int, ...],
    edge_lookup: dict[tuple[int, int], float],
    state_cost: np.ndarray,
) -> float:
    if not route:
        return math.inf
    return float(
        sum(
            edge_lookup[(min(left, right), max(left, right))]
            + float(state_cost[right])
            for left, right in pairwise(route)
        )
    )


def evaluate_lambda_selection(
    config: LambdaSelectionConfig,
    *,
    root: Path | None = None,
) -> LambdaSelectionReport:
    world_config = ProceduralEvaluationConfig(
        world_seeds=config.world_seeds,
        noise_seeds=(0,),
    )
    runs = []
    for seed in config.world_seeds:
        world = build_procedural_world(seed, world_config)
        cost = np.asarray(
            world.dynamic_cost[world.rescue_start_step],
            dtype=np.float64,
        )
        edge_lookup = {
            (min(int(left), int(right)), max(int(left), int(right))): float(weight)
            for (left, right), weight in zip(
                world.graph.edges,
                world.graph.edge_cost,
                strict=True,
            )
        }
        optimum = tuple(
            _route_objective(
                _route_to_any(
                    world.graph,
                    survivor,
                    world.exit_nodes,
                    cost,
                    astar=False,
                ),
                edge_lookup,
                cost,
            )
            for survivor in world.survivor_nodes
        )
        for control_temperature in config.candidates:
            regrets = []
            optimism_mean = []
            optimism_maximum = []
            sweeps = []
            converged = []
            for survivor, best in zip(
                world.survivor_nodes,
                optimum,
                strict=True,
            ):
                model = build_lmdp(
                    world.graph,
                    cost,
                    world.exit_nodes,
                    config=LMDPConfig(
                        control_temperature=control_temperature,
                        self_loop_weight=0.0,
                        minimum_state_cost=0.02,
                        tolerance=1.0e-8,
                        max_sweeps=10_000,
                    ),
                )
                solution = solve_jacobi(model)
                route = tuple(greedy_route(model, solution, survivor))
                objective = _route_objective(route, edge_lookup, cost)
                regrets.append(max(0.0, objective - best))
                diagnostic = trajectory_optimism(model, solution)
                optimism_mean.append(diagnostic.mean_gap)
                optimism_maximum.append(diagnostic.maximum_gap)
                sweeps.append(solution.sweeps)
                converged.append(solution.converged)
            runs.append(
                LambdaRun(
                    world_seed=seed,
                    control_temperature=control_temperature,
                    survivor_count=len(world.survivor_nodes),
                    mean_static_regret=float(np.mean(regrets)),
                    maximum_static_regret=float(np.max(regrets)),
                    mean_optimism_gap=float(np.mean(optimism_mean)),
                    maximum_optimism_gap=float(np.max(optimism_maximum)),
                    mean_solver_sweeps=float(np.mean(sweeps)),
                    all_converged=all(converged),
                )
            )

    run_tuple = tuple(runs)
    candidates = {}
    for value in config.candidates:
        selected = tuple(
            run for run in run_tuple if run.control_temperature == value
        )
        candidates[str(value)] = {
            "world_count": len(selected),
            "mean_static_regret": float(
                np.mean([run.mean_static_regret for run in selected])
            ),
            "maximum_static_regret": float(
                np.max([run.maximum_static_regret for run in selected])
            ),
            "mean_optimism_gap": float(
                np.mean([run.mean_optimism_gap for run in selected])
            ),
            "maximum_optimism_gap": float(
                np.max([run.maximum_optimism_gap for run in selected])
            ),
            "mean_solver_sweeps": float(
                np.mean([run.mean_solver_sweeps for run in selected])
            ),
            "all_converged": all(run.all_converged for run in selected),
        }
    selected_lambda = min(
        config.candidates,
        key=lambda value: (
            candidates[str(value)]["mean_static_regret"],
            candidates[str(value)]["mean_optimism_gap"],
            value,
        ),
    )
    summary = {
        "world_count": len(config.world_seeds),
        "candidate_count": len(config.candidates),
        "selected_lambda": selected_lambda,
        "selection_rule": "minimum_mean_static_regret_then_optimism_gap",
        "candidates": candidates,
        "claim_checks": {
            "all_candidates_converged": all(
                item["all_converged"] for item in candidates.values()
            ),
            "development_selection_complete": len(config.world_seeds) >= 20,
            "held_out_seeds_opened": config.split == "held_out",
            "paper_claim_ready": False,
        },
    }
    protocol = {
        "split": config.split,
        "world_seeds": list(config.world_seeds),
        "candidates": list(config.candidates),
        "objective": "static_clairvoyant_route_regret",
        "tie_breaker": "trajectory_optimism_gap",
        "selection_is_frozen_for_future_held_out_use": False,
    }
    manifest = capture_manifest(
        config.world_seeds[0],
        protocol,
        {"run_count": len(run_tuple)},
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
    return LambdaSelectionReport(
        schema_version=1,
        protocol=protocol,
        provenance=provenance,
        runs=run_tuple,
        summary=summary,
    )


def write_report(report: LambdaSelectionReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report.to_json(), encoding="utf-8")
    temporary.replace(path)


def format_summary(report: LambdaSelectionReport) -> str:
    selected = report.summary["selected_lambda"]
    metrics = report.summary["candidates"][str(selected)]
    return "\n".join(
        (
            f"lambda selection worlds: {report.summary['world_count']}",
            f"selected lambda: {selected}",
            f"mean static regret: {metrics['mean_static_regret']:.6f}",
            f"mean optimism gap: {metrics['mean_optimism_gap']:.6f}",
            "held-out seeds: SEALED",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-seeds", default="0:19")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/eval/lambda-selection.json"),
    )
    parser.add_argument("--allow-held-out", action="store_true")
    args = parser.parse_args(argv)
    report = evaluate_lambda_selection(
        LambdaSelectionConfig(
            world_seeds=parse_seed_spec(args.world_seeds),
            allow_held_out=args.allow_held_out,
        )
    )
    write_report(report, args.out)
    print(format_summary(report))
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
