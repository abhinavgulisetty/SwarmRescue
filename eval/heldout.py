"""Hash-locked held-out evaluation for the frozen Phase 3 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ember.manifest import capture_manifest
from eval.procedural import (
    HELD_OUT_WORLD_SEEDS,
    ProceduralEvaluationConfig,
    ProceduralEvaluationReport,
    evaluate_procedural,
    format_procedural_summary,
    write_procedural_csv,
    write_procedural_report,
)

DEFAULT_PROTOCOL = Path(__file__).with_name("frozen_protocol.json")
FROZEN_PROTOCOL_SHA256 = (
    "e77caa47310419c8329f5d66c31dfd5b523822ddde21f743d647f28a578d647c"
)


def _canonical_bytes(protocol: dict[str, Any]) -> bytes:
    return json.dumps(
        protocol,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def load_frozen_protocol(path: Path = DEFAULT_PROTOCOL) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(_canonical_bytes(protocol)).hexdigest()
    if digest != FROZEN_PROTOCOL_SHA256:
        raise ValueError(
            "frozen protocol hash mismatch; do not change evaluation choices "
            "after held-out access"
        )
    if protocol.get("schema_version") != 1:
        raise ValueError("unsupported frozen protocol schema")
    if protocol.get("status") != "frozen_before_held_out":
        raise ValueError("protocol was not frozen before held-out access")
    evaluation = protocol.get("evaluation", {})
    if (
        evaluation.get("world_seed_first") != min(HELD_OUT_WORLD_SEEDS)
        or evaluation.get("world_seed_last") != max(HELD_OUT_WORLD_SEEDS)
    ):
        raise ValueError("frozen protocol does not cover the complete held-out set")
    return protocol, digest


def evaluate_frozen_heldout(
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    root: Path | None = None,
) -> ProceduralEvaluationReport:
    frozen, digest = load_frozen_protocol(protocol_path)
    values = frozen["evaluation"]
    config = ProceduralEvaluationConfig(
        world_seeds=tuple(
            range(values["world_seed_first"], values["world_seed_last"] + 1)
        ),
        noise_seeds=tuple(values["noise_seeds"]),
        bootstrap_samples=values["bootstrap_samples"],
        confidence_level=values["confidence_level"],
        bootstrap_seed=values["bootstrap_seed"],
        robot_count=values["robot_count"],
        survivor_count=values["survivor_count"],
        hazard_horizon_s=values["hazard_horizon_s"],
        voi_packet_budget_bytes=values["voi_packet_budget_bytes"],
        lmdp_control_temperature=values["lmdp_control_temperature"],
        uncertainty_beta=values["uncertainty_beta"],
        allow_held_out=True,
        communication_policy="adjoint_kl",
        inference_policy="gbp",
        escort_policy="lmdp",
    )
    report = evaluate_procedural(config, root=root)
    report_protocol = {
        **report.protocol,
        "evaluation_phase": "historical_held_out_replay",
        "held_out_previously_opened": True,
        "implementation_locked_by_protocol": False,
        "fresh_confirmatory_evidence": False,
        "frozen_protocol_sha256": digest,
        "claim_gates": frozen["claim_gates"],
        "predeclared_validity_boundaries": frozen["validity_boundaries"],
    }
    manifest = capture_manifest(
        config.world_seeds[0],
        report_protocol,
        {"scenario_count": len(report.runs), "world_count": len(config.world_seeds)},
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
    return replace(report, protocol=report_protocol, provenance=provenance)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/eval/heldout-replay.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/eval/heldout-replay.csv"),
    )
    args = parser.parse_args(argv)
    report = evaluate_frozen_heldout(protocol_path=args.protocol)
    write_procedural_report(report, args.out)
    write_procedural_csv(report, args.csv)
    print(format_procedural_summary(report))
    print(f"protocol sha256: {report.protocol['frozen_protocol_sha256']}")
    print(f"report: {args.out}")
    print(f"runs:   {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
