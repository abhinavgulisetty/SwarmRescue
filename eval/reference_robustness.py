"""Fixed no-path reference robustness check on already evaluated frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eval.constrained_benchmark import make_case
from eval.reference_constrained import reference_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--primary-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.absolute()
    primary = json.loads(args.primary_report.read_text())
    protocol = json.loads(
        Path(__file__).with_name("constrained_protocol.json").read_text()
    )
    root = Path(__file__).resolve().parents[1]

    def unchanged():
        return all(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
            for name, digest in protocol["source_files_sha256"].items()
        )

    assert primary["split"] == "evaluation" and unchanged()
    receipt = {
        "configuration": (
            "Upstream default make main; no -DPATH; integer bucket; no tie breaking"
        ),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "scope": (
            "Fixed documented default-build check on previously opened "
            "evaluation inputs; no candidate retuning or independent holdout claim."
        ),
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Record the fixed build before running any case.
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    for row in primary["cases"]:
        result = reference_call(binary, make_case(row["seed"], row["family"]))
        optimum = row["optimum_q16"]
        result["correct"] = result["status"] == "ok" and (
            result["feasible"] == (optimum is not None)
            and (optimum is None or result["cost_q16"] == optimum)
        )
        receipt["cases"].append(
            dict(
                seed=row["seed"],
                family=row["family"],
                optimum_q16=optimum,
                result=result,
            )
        )
    assert unchanged()
    receipt["correct"] = sum(row["result"]["correct"] for row in receipt["cases"])
    receipt["total"] = len(receipt["cases"])
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Default reference correct: {receipt['correct']}/{receipt['total']}")


if __name__ == "__main__":
    main()
