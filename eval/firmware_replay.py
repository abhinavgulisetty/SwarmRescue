"""Host and Cortex-M4 QEMU replay of independent integer-kernel fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cascade.extract import TraversabilityGraph, VoxelGrid
from ember.manifest import capture_manifest, write_runtime_source_bundle
from ember.routing import LMDPConfig, LMDPModel
from ember.routing.deadline import deadline_route
from eval.harness import parse_seed_spec
from mcu import INT32_MAX, Q16_SCALE
from mcu.log_lmdp import solve_log_q16

ROOT = Path(__file__).resolve().parents[1]


def generate_header(seeds):
    parts = ["#include <stdint.h>"]
    logs, deadlines = [], []
    max_nodes = 500

    def array(name, values, dtype="int32_t"):
        encoded = ",".join(str(int(value)) for value in values)
        parts.append(f"static const {dtype} {name}[] = {{{encoded or '0'}}};")
        return name

    chain = LMDPModel(
        np.arange(501),
        np.minimum(np.arange(500) + 1, 499),
        np.ones(500),
        np.full(500, 2.0),
        np.arange(500) == 499,
        np.zeros(500, bool),
        LMDPConfig(max_sweeps=510),
    )
    boundary = LMDPModel(
        np.arange(6),
        np.array([1, 1, 3, 3, 4]),
        np.ones(5),
        np.ones(5),
        np.array([0, 1, 0, 0, 0], bool),
        np.array([0, 0, 0, 1, 0], bool),
        LMDPConfig(),
    )
    models = [chain, boundary]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        n = 12
        probabilities = rng.uniform(0.01, 1.0, (n, n))
        probabilities /= probabilities.sum(axis=1)[:, None]
        models.append(
            LMDPModel(
                np.arange(n + 1) * n,
                np.tile(np.arange(n), n),
                probabilities.ravel(),
                rng.uniform(0.2, 4.0, n),
                np.arange(n) == n - 1,
                np.arange(n) == n - 2,
                LMDPConfig(max_sweeps=4000),
            )
        )
    for index, model in enumerate(models):
        solution = solve_log_q16(model)
        if not solution.fixed_point:
            raise RuntimeError("reference did not converge")
        q = solution.model
        values = solution.negative_log_desirability
        actions = []
        for state in range(q.n_states):
            candidates = []
            if values[state] != INT32_MAX:
                for edge in range(int(q.row_ptr[state]), int(q.row_ptr[state + 1])):
                    node = int(q.col_idx[edge])
                    passive = int(q.negative_log_passive[edge])
                    if passive != INT32_MAX and values[node] < values[state]:
                        candidates.append((int(values[node]) + passive, node))
            actions.append(min(candidates)[1] if candidates else -1)
        arrays = [
            array(f"l{index}_{name}", data, dtype)
            for name, data, dtype in (
                ("rows", q.row_ptr, "int32_t"),
                ("columns", q.col_idx, "int32_t"),
                ("passive", q.negative_log_passive, "int32_t"),
                ("cost", q.cost_over_temperature, "int32_t"),
                ("exits", q.exit_mask, "uint8_t"),
                ("deaths", q.death_mask, "uint8_t"),
                ("expected", values, "int32_t"),
                ("actions", actions, "int32_t"),
            )
        ]
        logs.append(
            "{"
            + ",".join(map(str, [q.n_states, len(q.col_idx), model.config.max_sweeps]))
            + ","
            + ",".join(arrays)
            + "}"
        )
    for seed in seeds:
        rng = np.random.default_rng(seed + 8192)
        n = 6
        occupied = np.zeros((n, 1, 1), bool)
        edges = np.array([[0, 1], [1, 2], [2, 5], [0, 3], [3, 4], [4, 5], [1, 4]])
        lengths = rng.uniform(0.4, 2.5, len(edges))
        risk = rng.uniform(0, 6, n)
        graph = TraversabilityGraph(
            VoxelGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), occupied, occupied),
            np.array([[i, 0, 0] for i in range(n)]),
            np.zeros(n),
            edges,
            lengths,
        )
        adjacency = [[] for _ in range(n)]
        for (left, right), length in zip(edges, lengths, strict=True):
            steps = max(1, math.ceil(float(length) / 0.8 / 0.5))
            for source, target in ((left, right), (right, left)):
                cost = round((float(length) + risk[target] * steps * 0.5) * Q16_SCALE)
                adjacency[source].append((target, steps, cost))
        rows = np.array([0, *np.cumsum([len(row) for row in adjacency])])
        flat = np.array([edge for row in adjacency for edge in row])
        exits = np.arange(n) == 5
        for capacity in (1, 2, 64):
            horizon = int(rng.integers(1, 40))
            expected = deadline_route(
                graph,
                0,
                (5,),
                risk,
                available_steps=horizon,
                walking_speed_m_s=0.8,
                dt_s=0.5,
                max_labels=capacity,
            )
            index = len(deadlines)
            arrays = [
                array(f"d{index}_{name}", data, dtype)
                for name, data, dtype in (
                    ("rows", rows, "int32_t"),
                    ("columns", flat[:, 0], "int32_t"),
                    ("times", flat[:, 1], "int32_t"),
                    ("cost", flat[:, 2], "int32_t"),
                    ("exits", exits, "uint8_t"),
                    ("route", expected.route, "int32_t"),
                )
            ]
            fields = [
                n,
                len(flat),
                horizon,
                capacity,
                expected.travel_steps,
                len(expected.route),
                int(expected.feasible),
                int(expected.optimal),
                int(expected.resource_fallback),
                expected.labels_used,
            ]
            objective = round(expected.estimated_objective * Q16_SCALE)
            deadlines.append(
                "{"
                + ",".join(map(str, fields))
                + f",{objective}LL,"
                + ",".join(arrays)
                + "}"
            )
    parts.extend(
        [
            "typedef struct { int32_t n,edges,max_sweeps; "
            "const int32_t *rows,*columns,*passive,*cost; "
            "const uint8_t *exits,*deaths; "
            "const int32_t *expected,*actions; } log_fixture;",
            "typedef struct { int32_t n,edges,deadline,capacity,time,route_length,"
            "feasible,optimal,fallback,labels_used; int64_t objective; "
            "const int32_t *rows,*columns,*times,*cost; const uint8_t *exits; "
            "const int32_t *route; } deadline_fixture;",
            "static const log_fixture log_cases[] = {" + ",".join(logs) + "};",
            "static const deadline_fixture deadline_cases[] = {"
            + ",".join(deadlines)
            + "};",
            f"#define REPLAY_LOG_CASES {len(logs)}",
            f"#define REPLAY_DEADLINE_CASES {len(deadlines)}",
            f"#define REPLAY_MAX_N {max_nodes}",
            "#define REPLAY_MAX_LABELS 64",
        ]
    )
    return "\n".join(parts) + "\n"


def run_replay(build_dir, seeds, *, tool_root=None, arm=False):
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    fixture = build_dir / "fixtures.h"
    fixture.write_text(generate_header(seeds))
    environment = os.environ.copy()
    if tool_root:
        tool_root = Path(tool_root)
        environment["PATH"] = (
            str(tool_root / "usr/bin") + os.pathsep + environment.get("PATH", "")
        )
        environment["LD_LIBRARY_PATH"] = str(tool_root / "usr/lib/x86_64-linux-gnu")
    executable = build_dir / ("replay.elf" if arm else "replay-host")
    compiler = shutil.which(
        "arm-none-eabi-gcc" if arm else "cc", path=environment.get("PATH")
    )
    if not compiler:
        raise RuntimeError("required compiler is unavailable")
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(ROOT / "firmware"),
        "-I",
        str(build_dir),
        str(ROOT / "firmware/replay/replay.c"),
        str(ROOT / "firmware/log_lmdp.c"),
        str(ROOT / "firmware/deadline.c"),
    ]
    if arm:
        command += [
            "-mcpu=cortex-m4",
            "-mthumb",
            "-mfloat-abi=soft",
            "-ffreestanding",
            "-fno-builtin",
            "-fno-common",
            "-nostdlib",
            "-Wl,-T," + str(ROOT / "firmware/replay/mps2.ld"),
            "-Wl,-Map," + str(build_dir / "replay.map"),
            "-lgcc",
        ]
    command += ["-o", str(executable)]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    if arm:
        qemu = shutil.which("qemu-system-arm", path=environment.get("PATH"))
        if not qemu:
            raise RuntimeError("QEMU is unavailable")
        command = [
            qemu,
            "-M",
            "mps2-an386",
            "-cpu",
            "cortex-m4",
            "-nographic",
            "-semihosting-config",
            "enable=on,target=native",
            "-kernel",
            str(executable),
        ]
    else:
        command = [str(executable)]
    result = subprocess.run(
        command, capture_output=True, text=True, env=environment, timeout=120
    )
    lines = [
        line
        for line in (result.stdout + "\n" + result.stderr).splitlines()
        if line.startswith("{")
    ]
    if not lines:
        raise RuntimeError("guest emitted no result: " + result.stderr[-2000:])
    metrics = json.loads(lines[-1])
    if result.returncode or metrics["failures"]:
        raise RuntimeError("kernel replay failed: " + json.dumps(metrics))
    metrics["executable_sha256"] = hashlib.sha256(executable.read_bytes()).hexdigest()
    metrics["fixture_sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
    metrics["compiler"] = subprocess.run(
        [compiler, "--version"],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    ).stdout.splitlines()[0]
    if arm:
        metrics["emulator"] = subprocess.run(
            [qemu, "--version"],
            capture_output=True,
            text=True,
            check=True,
            env=environment,
        ).stdout.splitlines()[0]
        size = shutil.which("arm-none-eabi-size", path=environment.get("PATH"))
        sizes = (
            subprocess.run(
                [size, str(executable)],
                capture_output=True,
                text=True,
                check=True,
                env=environment,
            )
            .stdout.splitlines()[1]
            .split()
        )
        (
            metrics["text_bytes"],
            metrics["data_bytes"],
            metrics["bss_and_reserved_stack_bytes"],
        ) = map(int, sizes[:3])
        metrics["stack_reserved_bytes"] = 16 * 1024
        metrics["kernel_harness_ram_with_reserved_stack_bytes"] = int(sizes[1]) + int(
            sizes[2]
        )
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="0:19")
    parser.add_argument("--tool-root", type=Path)
    parser.add_argument("--arm", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("outputs/eval/firmware-replay")
    )
    args = parser.parse_args(argv)
    seeds = parse_seed_spec(args.seeds)
    if not seeds or any(seed not in range(200) for seed in seeds):
        parser.error("only development fixture seeds 0..199 are permitted")
    config = {
        "seeds": seeds,
        "arm": args.arm,
        "board": "mps2-an386" if args.arm else "host",
    }
    before = capture_manifest(0, config, {})
    try:
        metrics = run_replay(args.out, seeds, tool_root=args.tool_root, arm=args.arm)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(error.stderr) from error
    after = capture_manifest(0, config, metrics)
    if before.source_identity != after.source_identity:
        raise RuntimeError("source changed during firmware validation")
    report = {
        "schema_version": 1,
        "config": config,
        "metrics": metrics,
        "provenance": asdict(after),
        "hardware_cycles_measured": False,
        "whole_embedded_memory_gate_met": False,
        "whole_phase1_acceptance_gate_met": False,
    }
    path = args.out / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    write_runtime_source_bundle(path.with_suffix(".source.tar.gz"))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
