"""External published NWRCA* comparator; no upstream code is bundled."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from eval.constrained_benchmark import (
    DEVELOPMENT_SEEDS,
    EVALUATION_SEEDS,
    FAMILIES,
    arcs,
    dp_optimum,
    make_case,
)

UPSTREAM = "https://bitbucket.org/s-ahmadi/multiobj"
REVISION = "4098543d0c5d78afc977017b5abf6054671a8c43"


def reference_call(binary, case):
    graph, start, goals, risk, deadline = case
    adjacent = arcs(graph, risk)
    edges = [
        (u, v, cost, steps) for u, row in enumerate(adjacent) for v, steps, cost in row
    ]
    # Positive constant terminal translation avoids unsupported zero-weight
    # bucket initialization. Every feasible route pays exactly one extra unit
    # of time and cost; budget translation preserves the original optimum.
    terminal = graph.n_nodes
    edges.extend((g, terminal, 1, 1) for g in goals)
    environment = os.environ.copy()
    environment["EMBER_REFERENCE_DEADLINE"] = str(deadline + 1)
    with tempfile.TemporaryDirectory(prefix="ember-nwrca-") as folder:
        input_path = Path(folder) / "matched.xy"
        text = [f"nodes {terminal + 1} edges {len(edges)} objectives 2"]
        text.extend(f"v {node} 0 0 0" for node in range(terminal + 1))
        text.extend(f"e {u} {v} {cost} {steps}" for u, v, cost, steps in edges)
        input_path.write_text("\n".join(text) + "\n")
        command = [
            str(binary),
            "--alg",
            "NWRCA",
            "--input",
            str(input_path),
            "--start",
            str(start),
            "--goal",
            str(terminal),
            "--constraint",
            "50",
            "--queue",
            "bucket",
            "--noheader",
            "--nruns",
            "3",
        ]
        begun = time.perf_counter_ns()
        try:
            process = subprocess.run(
                command,
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "returncode": None,
                "core_samples_ms": [],
                "feasible": None,
                "cost_q16": None,
            }
        wall_ms = (time.perf_counter_ns() - begun) / 1e6
    lines = [
        line.split()
        for line in process.stdout.splitlines()
        if line.startswith("NWRCA ")
    ]
    result = {
        "returncode": process.returncode,
        "wall_ms_three_runs": wall_ms,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "core_samples_ms": [],
        "feasible": None,
        "cost_q16": None,
    }
    if process.returncode != 0 or len(lines) != 3:
        return dict(result, status="process_failure")
    parsed = [
        (int(row[6]), int(row[7]), 1000 * (float(row[8]) + float(row[9])))
        for row in lines
    ]
    if len({(cost, count) for cost, count, _ in parsed}) != 1:
        return dict(result, status="inconsistent_repeats")
    cost, count, _ = parsed[0]
    result.update(
        status="ok",
        feasible=count > 0,
        cost_q16=cost - 1 if count else None,
        core_samples_ms=[row[2] for row in parsed],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.absolute()
    protocol_path = Path(__file__).with_name("constrained_protocol.json")
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else None
    if args.split == "evaluation":
        if not protocol or not protocol["frozen"]:
            raise RuntimeError("published comparison requires frozen evaluation")
        root = Path(__file__).resolve().parents[1]
        if any(
            hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
            for name, digest in protocol["source_files_sha256"].items()
        ):
            raise RuntimeError("sources differ from frozen evaluation")
        if (
            hashlib.sha256(binary.read_bytes()).hexdigest()
            != protocol["reference"]["binary_sha256"]
        ):
            raise RuntimeError("reference binary differs from frozen build")
    seeds = DEVELOPMENT_SEEDS if args.split == "development" else EVALUATION_SEEDS
    records = []
    for seed in seeds:
        for family in FAMILIES:
            case = make_case(seed, family)
            graph, start, goals, risk, deadline = case
            optimum = dp_optimum(arcs(graph, risk), start, goals, deadline)
            result = reference_call(binary, case)
            result["correct"] = result["status"] == "ok" and (
                result["feasible"] == (optimum is not None)
                and (optimum is None or result["cost_q16"] == optimum)
            )
            records.append(
                dict(
                    seed=seed,
                    family=family,
                    states=graph.n_nodes,
                    deadline=deadline,
                    optimum_q16=optimum,
                    result=result,
                )
            )
        print(f"Reference completed {seed}", flush=True)
    report = {
        "split": args.split,
        "upstream": UPSTREAM,
        "revision": REVISION,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "configuration": (
            "DIM=2; int32 cost; GCC -O3 -DPATH; bucket; "
            "no tie breaking; no path printer"
        ),
        "adaptation": "explicit integer budget; positive one-unit terminal translation",
        "timing_scope": (
            "author-reported initialization plus search; "
            "parsing, IPC and input serialization excluded"
        ),
        "memory_scope": "unbounded native memory, not a 200KiB whole-MCU comparator",
        "correct": sum(row["result"]["correct"] for row in records),
        "total": len(records),
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Correct: {report['correct']}/{report['total']}", flush=True)


if __name__ == "__main__":
    main()
