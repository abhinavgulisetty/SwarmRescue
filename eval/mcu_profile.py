"""Analytical Tier-A MCU resource profile over procedural development worlds.

This is not emulator evidence. It allocates every persistent Q16/NCA/GBP buffer in the
fixed arena and charges a documented conservative operation model against a 10 Hz,
240 MHz control-tick budget.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ember.comms import MAX_PAYLOAD_BYTES
from ember.manifest import capture_manifest
from ember.routing import Int8NCAWeights, LMDPConfig, build_lmdp
from eval.harness import parse_seed_spec
from eval.procedural import (
    DEVELOPMENT_WORLD_SEEDS,
    HELD_OUT_WORLD_SEEDS,
    ProceduralEvaluationConfig,
    build_procedural_world,
)
from mcu import CycleBudget, MCUArena


@dataclass(frozen=True, slots=True)
class MCUProfileConfig:
    world_seeds: tuple[int, ...] = tuple(range(20))
    clock_hz: int = 240_000_000
    control_rate_hz: int = 10
    ram_gate_bytes: int = 200 * 1024
    allow_held_out: bool = False

    def __post_init__(self) -> None:
        if not self.world_seeds:
            raise ValueError("world_seeds cannot be empty")
        if self.clock_hz <= 0 or self.control_rate_hz <= 0:
            raise ValueError("clock and control rate must be positive")
        if self.ram_gate_bytes <= 0:
            raise ValueError("ram gate must be positive")
        sealed = HELD_OUT_WORLD_SEEDS.intersection(self.world_seeds)
        if sealed and not self.allow_held_out:
            raise ValueError("held-out seeds are sealed until thresholds are frozen")

    @property
    def cycles_per_tick(self) -> int:
        return self.clock_hz // self.control_rate_hz

    @property
    def split(self) -> str:
        selected = set(self.world_seeds)
        if selected <= DEVELOPMENT_WORLD_SEEDS:
            return "development"
        if selected <= HELD_OUT_WORLD_SEEDS:
            return "held_out"
        return "custom"


@dataclass(frozen=True, slots=True)
class MCUProfileRun:
    world_seed: int
    states: int
    transitions: int
    undirected_edges: int
    nca_parameter_bytes: int
    arena_high_water_bytes: int
    ram_gate_bytes: int
    ram_gate_pass: bool
    analytical_cycles: int
    cycles_per_tick: int
    cycle_utilization: float
    deadline_miss: bool


@dataclass(frozen=True, slots=True)
class MCUProfileReport:
    schema_version: int
    protocol: dict[str, Any]
    provenance: dict[str, Any]
    runs: tuple[MCUProfileRun, ...]
    summary: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def _allocate_profile(
    states: int,
    transitions: int,
    undirected_edges: int,
    parameter_bytes: int,
) -> MCUArena:
    arena = MCUArena()
    allocations = (
        (states + 1) * 4,  # CSR row pointer
        transitions * 4,  # CSR column index
        transitions * 4,  # Q16 passive probabilities
        states * 4,  # Q16 attenuation
        states * 4,  # current desirability
        states * 4,  # next desirability
        states * 2,  # exit and death masks
        states * 8,  # local GBP information and precision
        2 * undirected_edges * 8,  # directed GBP information and precision
        parameter_bytes,
        4 * MAX_PAYLOAD_BYTES,  # bounded radio TX/RX ring
        1_024,  # scheduler, token bucket, and fixed control metadata
    )
    for size in allocations:
        arena.alloc(size, alignment=4)
    return arena


def _charge_profile(
    states: int,
    transitions: int,
    undirected_edges: int,
    limit: int,
) -> CycleBudget:
    budget = CycleBudget(limit)
    budget.charge(states, cycles_each=16)  # Bellman row setup and boundaries
    budget.charge(transitions, cycles_each=22)  # Q16 CSR multiply/accumulate
    budget.charge(states * (32 * 8 + 64 * 32 + 64), cycles_each=2)
    budget.charge(states, cycles_each=20)  # clipping and safeguard comparison
    budget.charge(transitions, cycles_each=22)  # safeguard residual Bellman pass
    budget.charge(2 * undirected_edges, cycles_each=36)  # one GBP message sweep
    budget.charge(states, cycles_each=24)  # belief/risk projection
    return budget


def evaluate_mcu_profile(
    config: MCUProfileConfig,
    *,
    root: Path | None = None,
) -> MCUProfileReport:
    world_config = ProceduralEvaluationConfig(
        world_seeds=config.world_seeds,
        noise_seeds=(0,),
    )
    weights = Int8NCAWeights.default()
    runs = []
    for seed in config.world_seeds:
        world = build_procedural_world(seed, world_config)
        model = build_lmdp(
            world.graph,
            np.asarray(
                world.dynamic_cost[world.rescue_start_step],
                dtype=np.float64,
            ),
            world.exit_nodes,
            config=LMDPConfig(tolerance=1.0e-8, max_sweeps=10_000),
        )
        states = model.n_states
        transitions = int(model.col_idx.size)
        undirected_edges = int(world.graph.edges.shape[0])
        arena = _allocate_profile(
            states,
            transitions,
            undirected_edges,
            weights.parameter_bytes,
        )
        budget = _charge_profile(
            states,
            transitions,
            undirected_edges,
            config.cycles_per_tick,
        )
        runs.append(
            MCUProfileRun(
                world_seed=seed,
                states=states,
                transitions=transitions,
                undirected_edges=undirected_edges,
                nca_parameter_bytes=weights.parameter_bytes,
                arena_high_water_bytes=arena.high_water,
                ram_gate_bytes=config.ram_gate_bytes,
                ram_gate_pass=arena.high_water < config.ram_gate_bytes,
                analytical_cycles=budget.used,
                cycles_per_tick=config.cycles_per_tick,
                cycle_utilization=budget.used / config.cycles_per_tick,
                deadline_miss=budget.missed,
            )
        )

    run_tuple = tuple(runs)
    ram = np.asarray([run.arena_high_water_bytes for run in run_tuple])
    utilization = np.asarray([run.cycle_utilization for run in run_tuple])
    deadline_rate = float(np.mean([run.deadline_miss for run in run_tuple]))
    summary = {
        "world_count": len(run_tuple),
        "maximum_arena_high_water_bytes": int(np.max(ram)),
        "mean_arena_high_water_bytes": float(np.mean(ram)),
        "maximum_cycle_utilization": float(np.max(utilization)),
        "mean_cycle_utilization": float(np.mean(utilization)),
        "deadline_miss_rate": deadline_rate,
        "claim_checks": {
            "ram_below_200_kib": all(run.ram_gate_pass for run in run_tuple),
            "deadline_miss_rate_below_5_percent": deadline_rate < 0.05,
            "host_c_bit_exact_test_available": True,
            "emulator_calibration_complete": False,
            "tier_b_complete": False,
        },
    }
    protocol = {
        "split": config.split,
        "world_seeds": list(config.world_seeds),
        "clock_hz": config.clock_hz,
        "control_rate_hz": config.control_rate_hz,
        "cycles_per_tick": config.cycles_per_tick,
        "ram_gate_bytes": config.ram_gate_bytes,
        "cycle_model": {
            "bellman_state": 16,
            "bellman_transition": 22,
            "nca_mac": 2,
            "nca_state_postprocess": 20,
            "safeguard_transition": 22,
            "gbp_directed_message": 36,
            "belief_state": 24,
        },
        "measurement_kind": "tier_a_analytical_not_emulator_measured",
        "component_benchmark_only": True,
        "held_out_seeds_opened": config.split == "held_out",
    }
    manifest = capture_manifest(
        config.world_seeds[0],
        protocol,
        {"world_count": len(run_tuple)},
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
    return MCUProfileReport(
        schema_version=1,
        protocol=protocol,
        provenance=provenance,
        runs=run_tuple,
        summary=summary,
    )


def write_report(report: MCUProfileReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report.to_json(), encoding="utf-8")
    temporary.replace(path)


def write_csv(report: MCUProfileReport, path: Path) -> None:
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


def format_summary(report: MCUProfileReport) -> str:
    summary = report.summary
    checks = summary["claim_checks"]
    return "\n".join(
        (
            f"MCU profile worlds: {summary['world_count']}",
            "maximum arena high-water: "
            f"{summary['maximum_arena_high_water_bytes']} bytes",
            "maximum analytical cycle utilization: "
            f"{summary['maximum_cycle_utilization']:.2%}",
            f"deadline-miss rate: {summary['deadline_miss_rate']:.2%}",
            "Tier-B emulator calibration: "
            f"{'READY' if checks['emulator_calibration_complete'] else 'NOT READY'}",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-seeds", default="0:19")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/eval/mcu-profile.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/eval/mcu-profile.csv"),
    )
    parser.add_argument("--allow-held-out", action="store_true")
    args = parser.parse_args(argv)
    report = evaluate_mcu_profile(
        MCUProfileConfig(
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
