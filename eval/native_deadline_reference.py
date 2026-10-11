"""External published default-build NWRCA* on the fresh native protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eval.constrained_benchmark import FAMILIES, arcs, dp_optimum, make_case
from eval.native_deadline_benchmark import (
    DEVELOPMENT_SEEDS,
    EVALUATION_SEEDS,
    source_hashes,
)
from eval.reference_constrained import reference_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "evaluation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.absolute()
    hashes = source_hashes()
    if args.split == "evaluation":
        protocol = json.loads(
            Path(__file__).with_name("native_deadline_protocol.json").read_text()
        )
        if not protocol["frozen"] or hashes != protocol["source_files_sha256"]:
            raise RuntimeError(
                "published comparison differs from sealed native protocol"
            )
        if (
            hashlib.sha256(binary.read_bytes()).hexdigest()
            != protocol["reference"]["binary_sha256"]
        ):
            raise RuntimeError("published reference binary differs from frozen build")
    seeds = DEVELOPMENT_SEEDS if args.split == "development" else EVALUATION_SEEDS
    rows = []
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
            rows.append(
                dict(seed=seed, family=family, optimum_q16=optimum, result=result)
            )
        print(f"Reference completed {seed}", flush=True)
    if source_hashes() != hashes:
        raise RuntimeError("published comparison sources changed")
    report = {
        "split": args.split,
        "upstream_revision": "4098543d0c5d78afc977017b5abf6054671a8c43",
        "configuration": (
            "DIM=2, int32, default make main -O3; bucket; "
            "no tie breaking; no path support"
        ),
        "timing_scope": (
            "Author initialization plus search, excluding parsing/IPC; "
            "not identical to full candidate API call."
        ),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "source_files_sha256": hashes,
        "cases": rows,
        "correct": sum(r["result"]["correct"] for r in rows),
        "total": len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Correct {report['correct']}/{report['total']}", flush=True)


if __name__ == "__main__":
    main()
