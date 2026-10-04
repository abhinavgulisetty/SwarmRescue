"""Paired development ablations for the EMBER procedural stack.

The grid isolates VoI scheduling, Gaussian belief propagation, the uncertainty
margin, the safeguarded NCA, and the Q16 fixed-point kernel.  It is a component
diagnostic: held-out seeds remain sealed and no result is promoted to a physical
mission claim.
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

from ember.gbp import GaussianBeliefPropagation, GBPConfig
from ember.manifest import capture_manifest
from ember.routing import (
    LMDPConfig,
    LMDPModel,
    LMDPSolution,
    build_lmdp,
    greedy_route,
    solve_jacobi,
    solve_safeguarded_nca,
)
from eval.harness import parse_seed_spec
from eval.procedural import (
    DEVELOPMENT_WORLD_SEEDS,
    HELD_OUT_WORLD_SEEDS,
    ProceduralEvaluationConfig,
    ProceduralWorld,
    _belief_from_packets,
    _route_to_any,
    build_procedural_world,
    sample_exogenous_streams,
)
from mcu import Q16_SCALE
from mcu.lmdp import q16_lmdp_sweep
from mcu.log_lmdp import solve_log_q16

ABLATION_VARIANTS = ("full", "no_voi", "no_gbp", "beta_zero", "nca")


@dataclass(frozen=True, slots=True)
class AblationConfig:
    world_seeds: tuple[int, ...] = tuple(range(20))
    noise_seed: int = 0
    control_temperature: float = 0.25
    uncertainty_beta: float = 1.0
    voi_packet_budget_bytes: int = 64
    tolerance: float = 1.0e-7
    max_sweeps: int = 10_000
    allow_held_out: bool = False

    def __post_init__(self) -> None:
        if not self.world_seeds:
            raise ValueError("world_seeds cannot be empty")
        if len(set(self.world_seeds)) != len(self.world_seeds):
            raise ValueError("world_seeds must be unique")
        if any(seed < 0 for seed in self.world_seeds):
            raise ValueError("world_seeds must be nonnegative")
        for name, value in (
            ("control_temperature", self.control_temperature),
            ("uncertainty_beta", self.uncertainty_beta),
            ("tolerance", self.tolerance),
        ):
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and positive")
        if self.voi_packet_budget_bytes < 20 or self.max_sweeps < 1:
            raise ValueError("packet budget and sweep cap are too small")
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
class AblationRun:
    world_seed: int
    variant: str
    radio_bytes: int
    inference_sweeps: int
    routing_sweeps: int
    mean_route_objective: float
    mean_static_regret: float
    all_routes_reached_exit: bool
    solver_converged: bool


@dataclass(frozen=True, slots=True)
class Q16ComparisonRun:
    world_seed: int
    float_sweeps: int
    q16_sweeps: int
    q16_fixed_point: bool
    maximum_desirability_error: float
    mean_desirability_error: float
    route_agreement_ratio: float
    route_completion_ratio: float


@dataclass(frozen=True, slots=True)
class LogQ16ComparisonRun:
    world_seed: int
    sweeps: int
    fixed_point: bool
    maximum_desirability_error: float
    maximum_value_error: float
    route_agreement_ratio: float
    route_completion_ratio: float
    arena_high_water_bytes: int


@dataclass(frozen=True, slots=True)
class AblationReport:
    schema_version: int
    protocol: dict[str, Any]
    provenance: dict[str, Any]
    runs: tuple[AblationRun, ...]
    q16_runs: tuple[Q16ComparisonRun, ...]
    log_q16_runs: tuple[LogQ16ComparisonRun, ...]
    summary: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def _gbp_belief(world: ProceduralWorld, packets) -> tuple[Any, int]:
    engine = GaussianBeliefPropagation(
        world.graph,
        mean=0.30,
        variance=0.04,
        config=GBPConfig(
            edge_variance=0.8,
            damping=0.6,
            tolerance=1.0e-6,
            max_sweeps=1_000,
        ),
    )
    for packet in packets:
        engine.observe_canonical(
            packet.cells.astype(np.int64),
            packet.information_q16.astype(np.float64) / Q16_SCALE,
            packet.precision_q16.astype(np.float64) / Q16_SCALE,
        )
    result = engine.run()
    if not result.converged:
        raise RuntimeError("ablation GBP did not converge")
    return result.belief, result.sweeps


def _model(
    world: ProceduralWorld,
    state_cost: np.ndarray,
    config: AblationConfig,
) -> LMDPModel:
    return build_lmdp(
        world.graph,
        state_cost,
        world.exit_nodes,
        config=LMDPConfig(
            control_temperature=config.control_temperature,
            self_loop_weight=0.0,
            minimum_state_cost=0.02,
            tolerance=config.tolerance,
            max_sweeps=config.max_sweeps,
        ),
    )


def _edge_lookup(world: ProceduralWorld) -> dict[tuple[int, int], float]:
    return {
        (min(int(left), int(right)), max(int(left), int(right))): float(weight)
        for (left, right), weight in zip(
            world.graph.edges,
            world.graph.edge_cost,
            strict=True,
        )
    }


def _route_objective(
    route: tuple[int, ...],
    edge_lookup: dict[tuple[int, int], float],
    truth: np.ndarray,
) -> float:
    if not route:
        return math.inf
    return float(
        sum(
            edge_lookup[(min(left, right), max(left, right))]
            + float(truth[right])
            for left, right in pairwise(route)
        )
    )


def _route_metrics(
    world: ProceduralWorld,
    model: LMDPModel,
    solution: LMDPSolution,
) -> tuple[float, float, bool, tuple[tuple[int, ...], ...]]:
    truth = np.asarray(
        world.dynamic_cost[world.rescue_start_step],
        dtype=np.float64,
    )
    lookup = _edge_lookup(world)
    routes = tuple(
        tuple(greedy_route(model, solution, survivor))
        for survivor in world.survivor_nodes
    )
    objectives = tuple(
        _route_objective(route, lookup, truth) for route in routes
    )
    optimum = tuple(
        _route_objective(
            _route_to_any(
                world.graph,
                survivor,
                world.exit_nodes,
                truth,
                astar=False,
            ),
            lookup,
            truth,
        )
        for survivor in world.survivor_nodes
    )
    reached = all(
        route and route[-1] in world.exit_nodes for route in routes
    )
    if not reached:
        raise RuntimeError("ablation route did not reach an exit")
    regret = [max(0.0, value - best) for value, best in zip(
        objectives, optimum, strict=True
    )]
    return float(np.mean(objectives)), float(np.mean(regret)), reached, routes


def _q16_probabilities(model: LMDPModel) -> np.ndarray:
    quantized = np.rint(model.passive_prob * Q16_SCALE).astype(np.int32)
    for state in range(model.n_states):
        row = model.row(state)
        local = quantized[row]
        correction = Q16_SCALE - int(np.sum(local, dtype=np.int64))
        local[int(np.argmax(model.passive_prob[row]))] += correction
    return quantized


def _q16_solution(
    model: LMDPModel,
) -> tuple[LMDPSolution, int, bool]:
    passive = _q16_probabilities(model)
    attenuation = np.rint(
        np.exp(-model.state_cost / model.config.control_temperature) * Q16_SCALE
    ).astype(np.int32)
    current = np.zeros(model.n_states, dtype=np.int32)
    current[model.exit_mask] = Q16_SCALE
    fixed = False
    sweeps = 0
    for sweep in range(1, model.config.max_sweeps + 1):
        sweeps = sweep
        following = q16_lmdp_sweep(
            model.row_ptr,
            model.col_idx,
            passive,
            attenuation,
            current,
            model.exit_mask,
            model.death_mask,
        )
        if np.array_equal(following, current):
            current = following
            fixed = True
            break
        current = following

    desirability = current.astype(np.float64) / Q16_SCALE
    value = np.full(model.n_states, np.inf)
    positive = desirability > 0.0
    value[positive] = (
        -model.config.control_temperature * np.log(desirability[positive])
    )
    value[model.exit_mask] = 0.0
    numerator = model.passive_prob * desirability[model.col_idx]
    denominator = np.add.reduceat(numerator, model.row_ptr[:-1])
    policy = np.zeros_like(numerator)
    for state in range(model.n_states):
        row = model.row(state)
        if not model.death_mask[state] and denominator[state] > 0.0:
            policy[row] = numerator[row] / denominator[state]
    return (
        LMDPSolution(
            desirability=desirability,
            value=value,
            policy_prob=policy,
            sweeps=sweeps,
            residual=0.0,
            relative_residual=0.0,
            converged=fixed,
        ),
        sweeps,
        fixed,
    )


def _run_variant(
    world: ProceduralWorld,
    variant: str,
    belief,
    inference_sweeps: int,
    radio_bytes: int,
    config: AblationConfig,
) -> tuple[AblationRun, LMDPModel, LMDPSolution, tuple[tuple[int, ...], ...]]:
    beta = 0.0 if variant == "beta_zero" else config.uncertainty_beta
    model = _model(world, belief.routing_cost(beta), config)
    if variant == "nca":
        solution = solve_safeguarded_nca(model).solution
    else:
        solution = solve_jacobi(model)
    objective, regret, reached, routes = _route_metrics(world, model, solution)
    return (
        AblationRun(
            world_seed=world.world_seed,
            variant=variant,
            radio_bytes=radio_bytes,
            inference_sweeps=inference_sweeps,
            routing_sweeps=solution.sweeps,
            mean_route_objective=objective,
            mean_static_regret=regret,
            all_routes_reached_exit=reached,
            solver_converged=solution.converged,
        ),
        model,
        solution,
        routes,
    )


def evaluate_ablations(
    config: AblationConfig,
    *,
    root: Path | None = None,
) -> AblationReport:
    world_config = ProceduralEvaluationConfig(
        world_seeds=config.world_seeds,
        noise_seeds=(config.noise_seed,),
        voi_packet_budget_bytes=config.voi_packet_budget_bytes,
        lmdp_control_temperature=config.control_temperature,
        allow_held_out=config.allow_held_out,
    )
    runs: list[AblationRun] = []
    q16_runs: list[Q16ComparisonRun] = []
    log_q16_runs: list[LogQ16ComparisonRun] = []

    for seed in config.world_seeds:
        world = build_procedural_world(seed, world_config)
        streams = sample_exogenous_streams(
            world,
            config.noise_seed,
            voi_packet_budget_bytes=config.voi_packet_budget_bytes,
            lmdp_control_temperature=config.control_temperature,
        )
        full_belief, full_inference = _gbp_belief(
            world, streams.voi_received_packets
        )
        flooding_belief, flooding_inference = _gbp_belief(
            world, streams.received_packets
        )
        direct_belief = _belief_from_packets(
            world, streams.voi_received_packets
        )

        full, full_model, full_solution, full_routes = _run_variant(
            world,
            "full",
            full_belief,
            full_inference,
            streams.voi_radio_stats.bytes_air,
            config,
        )
        runs.append(full)
        runs.append(
            _run_variant(
                world,
                "no_voi",
                flooding_belief,
                flooding_inference,
                streams.radio_stats.bytes_air,
                config,
            )[0]
        )
        runs.append(
            _run_variant(
                world,
                "no_gbp",
                direct_belief,
                0,
                streams.voi_radio_stats.bytes_air,
                config,
            )[0]
        )
        runs.append(
            _run_variant(
                world,
                "beta_zero",
                full_belief,
                full_inference,
                streams.voi_radio_stats.bytes_air,
                config,
            )[0]
        )
        runs.append(
            _run_variant(
                world,
                "nca",
                full_belief,
                full_inference,
                streams.voi_radio_stats.bytes_air,
                config,
            )[0]
        )

        log_solution = solve_log_q16(full_model)
        log_routes = tuple(
            log_solution.route(survivor) for survivor in world.survivor_nodes
        )
        log_difference = np.abs(
            full_solution.desirability - log_solution.desirability_reference()
        )
        finite = np.isfinite(full_solution.value)
        log_q16_runs.append(
            LogQ16ComparisonRun(
                world_seed=seed,
                sweeps=log_solution.sweeps,
                fixed_point=log_solution.fixed_point,
                maximum_desirability_error=float(np.max(log_difference)),
                maximum_value_error=float(np.max(np.abs(
                    full_solution.value[finite]
                    - log_solution.negative_log_desirability[finite]
                    / Q16_SCALE * full_model.config.control_temperature
                ))),
                route_agreement_ratio=float(np.mean([
                    left == right for left, right in zip(
                        full_routes, log_routes, strict=True
                    )
                ])),
                route_completion_ratio=float(np.mean([
                    bool(route) and route[-1] in world.exit_nodes
                    for route in log_routes
                ])),
                arena_high_water_bytes=log_solution.arena_high_water_bytes,
            )
        )

        q16_solution, q16_sweeps, q16_fixed = _q16_solution(full_model)
        q16_routes = tuple(
            tuple(greedy_route(full_model, q16_solution, survivor))
            for survivor in world.survivor_nodes
        )
        difference = np.abs(
            full_solution.desirability - q16_solution.desirability
        )
        q16_runs.append(
            Q16ComparisonRun(
                world_seed=seed,
                float_sweeps=full_solution.sweeps,
                q16_sweeps=q16_sweeps,
                q16_fixed_point=q16_fixed,
                maximum_desirability_error=float(np.max(difference)),
                mean_desirability_error=float(np.mean(difference)),
                route_agreement_ratio=float(
                    np.mean(
                        [
                            left == right
                            for left, right in zip(
                                full_routes, q16_routes, strict=True
                            )
                        ]
                    )
                ),
                route_completion_ratio=float(
                    np.mean(
                        [
                            bool(route) and route[-1] in world.exit_nodes
                            for route in q16_routes
                        ]
                    )
                ),
            )
        )

    run_tuple = tuple(runs)
    q16_tuple = tuple(q16_runs)
    variants: dict[str, Any] = {}
    for variant in ABLATION_VARIANTS:
        selected = tuple(run for run in run_tuple if run.variant == variant)
        variants[variant] = {
            "world_count": len(selected),
            "mean_route_objective": float(
                np.mean([run.mean_route_objective for run in selected])
            ),
            "mean_static_regret": float(
                np.mean([run.mean_static_regret for run in selected])
            ),
            "mean_radio_bytes": float(
                np.mean([run.radio_bytes for run in selected])
            ),
            "mean_inference_sweeps": float(
                np.mean([run.inference_sweeps for run in selected])
            ),
            "mean_routing_sweeps": float(
                np.mean([run.routing_sweeps for run in selected])
            ),
            "all_routes_reached_exit": all(
                run.all_routes_reached_exit for run in selected
            ),
            "all_solvers_converged": all(
                run.solver_converged for run in selected
            ),
        }
    full_metrics = variants["full"]
    contrasts = {
        variant: {
            "mean_objective_delta_from_full": (
                metrics["mean_route_objective"]
                - full_metrics["mean_route_objective"]
            ),
            "mean_radio_byte_delta_from_full": (
                metrics["mean_radio_bytes"] - full_metrics["mean_radio_bytes"]
            ),
        }
        for variant, metrics in variants.items()
        if variant != "full"
    }
    nca_speedup = (
        full_metrics["mean_routing_sweeps"]
        / variants["nca"]["mean_routing_sweeps"]
    )
    summary = {
        "world_count": len(config.world_seeds),
        "variants": variants,
        "paired_contrasts": contrasts,
        "nca_mean_sweep_speedup": float(nca_speedup),
        "q16": {
            "all_fixed_points_reached": all(run.q16_fixed_point for run in q16_tuple),
            "maximum_desirability_error": float(
                max(run.maximum_desirability_error for run in q16_tuple)
            ),
            "mean_desirability_error": float(
                np.mean([run.mean_desirability_error for run in q16_tuple])
            ),
            "mean_route_agreement_ratio": float(
                np.mean([run.route_agreement_ratio for run in q16_tuple])
            ),
            "mean_route_completion_ratio": float(
                np.mean([run.route_completion_ratio for run in q16_tuple])
            ),
        },
        "log_q16": {
            "all_fixed_points_reached": all(run.fixed_point for run in log_q16_runs),
            "maximum_desirability_error": max(
                run.maximum_desirability_error for run in log_q16_runs
            ),
            "maximum_value_error": max(
                run.maximum_value_error for run in log_q16_runs
            ),
            "mean_route_agreement_ratio": float(np.mean([
                run.route_agreement_ratio for run in log_q16_runs
            ])),
            "mean_route_completion_ratio": float(np.mean([
                run.route_completion_ratio for run in log_q16_runs
            ])),
            "maximum_arena_high_water_bytes": max(
                run.arena_high_water_bytes for run in log_q16_runs
            ),
        },
        "claim_checks": {
            "required_ablation_grid_complete": (
                len(config.world_seeds) >= 20
                and set(variants) == set(ABLATION_VARIANTS)
                and len(q16_tuple) == len(config.world_seeds)
            ),
            "held_out_seeds_opened": config.split == "held_out",
            "q16_route_gate_passed": all(
                run.route_completion_ratio == 1.0 for run in q16_tuple
            ),
            "log_q16_deployment_component_gate_passed": all(
                run.fixed_point and run.route_completion_ratio == 1.0
                and run.route_agreement_ratio == 1.0
                and run.arena_high_water_bytes < 200 * 1024
                for run in log_q16_runs
            ),
            "physical_mission_validated": False,
            "paper_claim_ready": False,
        },
    }
    protocol = {
        "split": config.split,
        "world_seeds": list(config.world_seeds),
        "noise_seed": config.noise_seed,
        "variants": list(ABLATION_VARIANTS),
        "paired_worlds_and_noise": True,
        "control_temperature": config.control_temperature,
        "uncertainty_beta": config.uncertainty_beta,
        "voi_packet_budget_bytes": config.voi_packet_budget_bytes,
        "q16_format": "Q16.16",
        "log_q16_format": "Q16.16_negative_log_desirability",
        "log_q16_lut_flash_bytes": 4097 * 4,
        "timing_is_hardware_measured": False,
        "component_ablation_only": True,
        "held_out_seeds_opened": config.split == "held_out",
    }
    manifest = capture_manifest(
        config.world_seeds[0],
        protocol,
        {"run_count": len(run_tuple), "q16_run_count": len(q16_tuple)},
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
    return AblationReport(
        schema_version=2,
        protocol=protocol,
        provenance=provenance,
        runs=run_tuple,
        q16_runs=q16_tuple,
        log_q16_runs=tuple(log_q16_runs),
        summary=summary,
    )


def write_report(report: AblationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report.to_json(), encoding="utf-8")
    temporary.replace(path)


def format_summary(report: AblationReport) -> str:
    summary = report.summary
    full = summary["variants"]["full"]
    no_voi = summary["variants"]["no_voi"]
    return "\n".join(
        (
            f"ablation worlds: {summary['world_count']}",
            f"full mean objective: {full['mean_route_objective']:.6f}",
            f"no-VoI mean objective: {no_voi['mean_route_objective']:.6f}",
            f"NCA sweep speedup: {summary['nca_mean_sweep_speedup']:.3f}x",
            (
                "Q16 max desirability error: "
                f"{summary['q16']['maximum_desirability_error']:.6g}"
            ),
            "held-out seeds: SEALED",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-seeds", default="0:19")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/eval/ablations.json"),
    )
    parser.add_argument("--allow-held-out", action="store_true")
    args = parser.parse_args(argv)
    report = evaluate_ablations(
        AblationConfig(
            world_seeds=parse_seed_spec(args.world_seeds),
            noise_seed=args.noise_seed,
            allow_held_out=args.allow_held_out,
        )
    )
    write_report(report, args.out)
    print(format_summary(report))
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
