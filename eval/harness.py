"""Paired multi-seed evaluation for the tactical EMBER reference mission.

Only implemented metrics are reported. Missing paper metrics are named
explicitly rather than replaced with proxies.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ember.baselines import BaselineResult
from ember.manifest import capture_manifest
from ember.mission import MissionResult, run_mission

DEVELOPMENT_SEEDS = frozenset(range(200))
HELD_OUT_SEEDS = frozenset(range(10_000, 10_050))
METHOD_ORDER = ("b1_shortest", "b2_flooding_dijkstra", "ember")
UNAVAILABLE_METRICS = (
    "iso_13571_survival",
    "rescue_time_s",
    "full_covariance_receiver_information_gain_nats",
    "ram_high_water_bytes",
    "mcu_cycles",
    "deadline_misses",
    "recovery_half_life_s",
    "b3_consensus_astar",
    "nca_sweeps",
    "procedural_scenario_diversity",
    "method_specific_communication_efficiency",
)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Immutable protocol settings shared by every paired method run."""

    seeds: tuple[int, ...] = tuple(range(200))
    bootstrap_samples: int = 2_000
    confidence_level: float = 0.95
    bootstrap_seed: int = 0xE4B3
    robot_count: int = 4
    observation_window_s: float = 1.6
    allow_held_out: bool = False

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in self.seeds
        ):
            raise ValueError("seeds must be nonnegative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.robot_count < 1:
            raise ValueError("robot_count must be positive")
        if self.observation_window_s <= 0.0 or not math.isfinite(
            self.observation_window_s
        ):
            raise ValueError("observation_window_s must be finite and positive")
        sealed = HELD_OUT_SEEDS.intersection(self.seeds)
        if sealed and not self.allow_held_out:
            raise ValueError(
                "held-out seeds 10000..10049 are sealed; pass allow_held_out=True "
                "only after implementation and thresholds are frozen"
            )

    @property
    def split(self) -> str:
        selected = set(self.seeds)
        if selected <= DEVELOPMENT_SEEDS:
            return "development"
        if selected <= HELD_OUT_SEEDS:
            return "held_out"
        return "custom"


@dataclass(frozen=True, slots=True)
class MethodRun:
    route: tuple[int, ...]
    route_dose: float
    path_length_m: float
    objective_cost: float
    clairvoyant_regret: float
    radio_bytes: int


@dataclass(frozen=True, slots=True)
class SeedRun:
    seed: int
    smoke_test_accepted: bool
    methods: dict[str, MethodRun]
    localization_error_m: float
    localization_radius_95_m: float
    localization_mahalanobis_squared: float
    localization_covered_95: bool
    receiver_information_gain_nats: float
    nats_per_byte: float
    radio_bytes: int
    radio_bytes_per_robot_s: float
    radio_deliveries: int
    gbp_sweeps: int
    routing_sweeps: int
    mission_wall_time_s: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    schema_version: int
    protocol: dict[str, Any]
    provenance: dict[str, Any]
    unavailable_metrics: tuple[str, ...]
    runs: tuple[SeedRun, ...]
    summary: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def _method_run(result: BaselineResult, oracle: BaselineResult) -> MethodRun:
    regret = result.objective_cost - oracle.objective_cost
    if regret < -1.0e-10:
        raise AssertionError("clairvoyant regret cannot be negative")
    return MethodRun(
        route=result.route,
        route_dose=result.hazard_dose,
        path_length_m=result.path_length,
        objective_cost=result.objective_cost,
        clairvoyant_regret=max(0.0, regret),
        radio_bytes=result.radio_bytes,
    )


def evaluate_seed(
    seed: int,
    config: EvaluationConfig,
    *,
    mission_runner: Callable[[int], MissionResult] = run_mission,
) -> SeedRun:
    """Run all policies on one common world, truth, and noise stream."""

    started = time.perf_counter()
    mission = mission_runner(seed)
    elapsed = time.perf_counter() - started
    oracle = mission.clairvoyant_baseline
    methods = {
        "b1_shortest": _method_run(mission.centralized_baseline, oracle),
        "b2_flooding_dijkstra": _method_run(mission.flooding_baseline, oracle),
        "ember": _method_run(mission.ember_result, oracle),
    }
    radio_bytes = mission.radio_bytes
    information_gain = mission.receiver_information_gain_nats
    return SeedRun(
        seed=seed,
        smoke_test_accepted=mission.succeeded,
        methods=methods,
        localization_error_m=mission.localization_error_m,
        localization_radius_95_m=mission.localization.uncertainty_radius_m,
        localization_mahalanobis_squared=(
            mission.localization.mahalanobis_squared(mission.survivor_position_m)
        ),
        localization_covered_95=(
            mission.localization.covers_95(mission.survivor_position_m)
        ),
        receiver_information_gain_nats=information_gain,
        nats_per_byte=information_gain / radio_bytes if radio_bytes else 0.0,
        radio_bytes=radio_bytes,
        radio_bytes_per_robot_s=(
            radio_bytes / config.robot_count / config.observation_window_s
        ),
        radio_deliveries=mission.radio_stats.delivered,
        gbp_sweeps=mission.gbp_sweeps,
        routing_sweeps=mission.routing_solution.sweeps,
        mission_wall_time_s=elapsed,
    )


def _bootstrap_mean_ci(
    values: np.ndarray,
    config: EvaluationConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size == 0 or not np.all(np.isfinite(sample)):
        raise ValueError("bootstrap input must be a nonempty finite vector")
    indices = rng.integers(
        0,
        sample.size,
        size=(config.bootstrap_samples, sample.size),
    )
    means = np.mean(sample[indices], axis=1)
    tail = (1.0 - config.confidence_level) / 2.0
    low, high = np.quantile(means, (tail, 1.0 - tail))
    return {
        "mean": float(np.mean(sample)),
        "minimum": float(np.min(sample)),
        "maximum": float(np.max(sample)),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def _method_summary(
    runs: Sequence[SeedRun],
    method: str,
    config: EvaluationConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    def values(attribute: str) -> np.ndarray:
        return np.asarray(
            [getattr(run.methods[method], attribute) for run in runs],
            dtype=np.float64,
        )

    return {
        "route_dose": _bootstrap_mean_ci(values("route_dose"), config, rng),
        "path_length_m": _bootstrap_mean_ci(values("path_length_m"), config, rng),
        "objective_cost": _bootstrap_mean_ci(values("objective_cost"), config, rng),
        "clairvoyant_regret": _bootstrap_mean_ci(
            values("clairvoyant_regret"), config, rng
        ),
        "radio_bytes": _bootstrap_mean_ci(values("radio_bytes"), config, rng),
    }


def _paired_summary(
    runs: Sequence[SeedRun],
    left: str,
    right: str,
    config: EvaluationConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    left_dose = np.asarray([run.methods[left].route_dose for run in runs])
    right_dose = np.asarray([run.methods[right].route_dose for run in runs])
    delta = left_dose - right_dose
    tolerance = 1.0e-12
    routes_equal = np.asarray(
        [run.methods[left].route == run.methods[right].route for run in runs]
    )
    result: dict[str, Any] = _bootstrap_mean_ci(delta, config, rng)
    result.update(
        {
            "metric": "route_dose",
            "left": left,
            "right": right,
            "wins": int(np.count_nonzero(delta < -tolerance)),
            "ties": int(np.count_nonzero(np.abs(delta) <= tolerance)),
            "losses": int(np.count_nonzero(delta > tolerance)),
            "same_route_rate": float(np.mean(routes_equal)),
        }
    )
    return result


def _summarize(
    runs: tuple[SeedRun, ...],
    config: EvaluationConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(config.bootstrap_seed)
    methods = {
        method: _method_summary(runs, method, config, rng) for method in METHOD_ORDER
    }
    ember_b2 = _paired_summary(runs, "ember", "b2_flooding_dijkstra", config, rng)
    ember_b1 = _paired_summary(runs, "ember", "b1_shortest", config, rng)
    system = {
        "smoke_test_acceptance_rate": float(
            np.mean([run.smoke_test_accepted for run in runs])
        ),
        "localization_coverage_95": float(
            np.mean([run.localization_covered_95 for run in runs])
        ),
        "localization_error_m": _bootstrap_mean_ci(
            np.asarray([run.localization_error_m for run in runs]), config, rng
        ),
        "localization_mahalanobis_squared": _bootstrap_mean_ci(
            np.asarray([run.localization_mahalanobis_squared for run in runs]),
            config,
            rng,
        ),
        "radio_bytes_per_robot_s": _bootstrap_mean_ci(
            np.asarray([run.radio_bytes_per_robot_s for run in runs]),
            config,
            rng,
        ),
        "receiver_information_gain_nats": _bootstrap_mean_ci(
            np.asarray([run.receiver_information_gain_nats for run in runs]),
            config,
            rng,
        ),
        "nats_per_byte": _bootstrap_mean_ci(
            np.asarray([run.nats_per_byte for run in runs]), config, rng
        ),
        "gbp_sweeps": _bootstrap_mean_ci(
            np.asarray([run.gbp_sweeps for run in runs]), config, rng
        ),
        "routing_sweeps": _bootstrap_mean_ci(
            np.asarray([run.routing_sweeps for run in runs]), config, rng
        ),
        "mission_wall_time_s": _bootstrap_mean_ci(
            np.asarray([run.mission_wall_time_s for run in runs]), config, rng
        ),
    }
    b2_superiority = bool(ember_b2["ci_high"] < 0.0)
    return {
        "methods": methods,
        "paired": {
            "ember_minus_b1": ember_b1,
            "ember_minus_b2": ember_b2,
        },
        "system": system,
        "claim_checks": {
            "ember_outperforms_b2_route_dose": b2_superiority,
            "b3_evaluated": False,
            "paper_superiority_claim_ready": False,
            "reason": (
                "B3 is not implemented; EMBER must also have a strictly "
                "negative paired confidence interval against B2."
                if b2_superiority
                else "EMBER does not show a strictly negative paired route-dose "
                "confidence interval against B2, and B3 is not implemented."
            ),
        },
    }


def evaluate(
    config: EvaluationConfig,
    *,
    mission_runner: Callable[[int], MissionResult] = run_mission,
    root: Path | None = None,
) -> EvaluationReport:
    """Evaluate paired policies and return a JSON-serializable report."""

    runs = tuple(
        evaluate_seed(seed, config, mission_runner=mission_runner)
        for seed in config.seeds
    )
    protocol = {
        "split": config.split,
        "scenario_family": "fixed_tactical_smoke_test",
        "world_varies_by_seed": False,
        "seed_varies": [
            "hazard_sensor_noise",
            "BLE_noise",
            "radio_packet_loss",
        ],
        "seeds": list(config.seeds),
        "paired_common_random_numbers": True,
        "communication_policy_varies_by_method": False,
        "communication_efficiency_comparison_available": False,
        "bootstrap_samples": config.bootstrap_samples,
        "confidence_level": config.confidence_level,
        "bootstrap_seed": config.bootstrap_seed,
        "robot_count": config.robot_count,
        "observation_window_s": config.observation_window_s,
        "localization_coverage_test": "mahalanobis_chi2_df2_0.95",
        "dose_is_heat_smoke_proxy_not_iso_13571": True,
        "receiver_information_is_marginal_gaussian_kl": True,
    }
    manifest = capture_manifest(
        config.seeds[0],
        protocol,
        {"seed_count": len(runs)},
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
    return EvaluationReport(
        schema_version=2,
        protocol=protocol,
        provenance=provenance,
        unavailable_metrics=UNAVAILABLE_METRICS,
        runs=runs,
        summary=_summarize(runs, config),
    )


def write_report(report: EvaluationReport, path: Path) -> None:
    """Atomically write the complete machine-readable report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report.to_json(), encoding="utf-8")
    temporary.replace(path)


def write_runs_csv(report: EvaluationReport, path: Path) -> None:
    """Write one tidy row per seed and method for external analysis."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = (
        "seed",
        "method",
        "route_dose",
        "path_length_m",
        "objective_cost",
        "clairvoyant_regret",
        "radio_bytes",
        "localization_error_m",
        "localization_covered_95",
        "localization_mahalanobis_squared",
        "nats_per_byte",
        "gbp_sweeps",
        "routing_sweeps",
    )
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in report.runs:
            for method in METHOD_ORDER:
                result = run.methods[method]
                writer.writerow(
                    {
                        "seed": run.seed,
                        "method": method,
                        "route_dose": result.route_dose,
                        "path_length_m": result.path_length_m,
                        "objective_cost": result.objective_cost,
                        "clairvoyant_regret": result.clairvoyant_regret,
                        "radio_bytes": result.radio_bytes,
                        "localization_error_m": run.localization_error_m,
                        "localization_covered_95": run.localization_covered_95,
                        "localization_mahalanobis_squared": (
                            run.localization_mahalanobis_squared
                        ),
                        "nats_per_byte": run.nats_per_byte,
                        "gbp_sweeps": run.gbp_sweeps,
                        "routing_sweeps": run.routing_sweeps,
                    }
                )
    temporary.replace(path)


def parse_seed_spec(specification: str) -> tuple[int, ...]:
    """Parse inclusive ranges such as 0:199 and comma-separated seeds."""

    seeds: list[int] = []
    for item in specification.split(","):
        token = item.strip()
        if not token:
            raise ValueError("empty seed item")
        if ":" in token:
            parts = token.split(":")
            if len(parts) != 2:
                raise ValueError(f"invalid seed range: {token}")
            start, stop = (int(value) for value in parts)
            if stop < start:
                raise ValueError("seed range stop must not precede start")
            seeds.extend(range(start, stop + 1))
        else:
            seeds.append(int(token))
    return tuple(seeds)


def format_summary(report: EvaluationReport) -> str:
    b2 = report.summary["paired"]["ember_minus_b2"]
    checks = report.summary["claim_checks"]
    return "\n".join(
        (
            f"paired seeds: {len(report.runs)} ({report.protocol['split']})",
            "EMBER - B2 dose: "
            f"{b2['mean']:.6f} "
            f"[{b2['ci_low']:.6f}, {b2['ci_high']:.6f}]",
            f"EMBER/B2 wins-ties-losses: {b2['wins']}-{b2['ties']}-{b2['losses']}",
            f"same-route rate: {b2['same_route_rate']:.1%}",
            "paper superiority claim: "
            + ("READY" if checks["paper_superiority_claim_ready"] else "NOT READY"),
            checks["reason"],
        )
    )


__all__ = [
    "DEVELOPMENT_SEEDS",
    "HELD_OUT_SEEDS",
    "EvaluationConfig",
    "EvaluationReport",
    "MethodRun",
    "SeedRun",
    "evaluate",
    "evaluate_seed",
    "format_summary",
    "parse_seed_spec",
    "write_report",
    "write_runs_csv",
]
