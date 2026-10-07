"""Reproducible software acceptance for complete physical rescue missions."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from ember.manifest import capture_manifest
from eval.harness import parse_seed_spec
from swarmrescue.rescue import main as rescue_main


def check_rescue_report(report):
    """Check current physical safe-zone witnesses, not just a success label."""
    metrics = report["metrics"]
    count = metrics["n_robots"]
    if not (
        report["accepted"]
        and report["physical_survivor_evacuation"]
        and report["rescue_acceptance_gate_met"]
        and metrics["survivors_hidden"]
        == metrics["survivors_contacted"]
        == metrics["survivors_evacuated"]
        == count
        >= 3
    ):
        raise ValueError("mission did not complete every physical rescue")
    if sorted(metrics["survivor_escort_robot"]) != list(range(count)):
        raise ValueError("escort ownership is not one-to-one")
    x0, y0, x1, y1 = metrics["evacuation_footprint_m"]
    for field, margin in (("survivor_positions_m", 0.5), ("robot_positions_m", 0.35)):
        positions = np.asarray(metrics[field], dtype=float)
        if positions.shape != (count, 3) or not np.all(np.isfinite(positions)):
            raise ValueError("physical coordinates are missing or nonfinite")
        depth = np.minimum.reduce(
            (
                positions[:, 0] - x0,
                x1 - positions[:, 0],
                positions[:, 1] - y0,
                y1 - positions[:, 1],
            )
        )
        if np.any(depth > -margin) or np.any(np.abs(positions[:, 2]) > 0.5):
            raise ValueError("physical bodies are not safely outside")
    for field in ("survivor_contact_time_s", "survivor_evacuation_time_s"):
        values = np.asarray(metrics[field], dtype=float)
        if values.shape != (count,) or not np.all(np.isfinite(values)):
            raise ValueError("rescue event times are missing")
    if np.any(
        np.asarray(metrics["survivor_evacuation_time_s"])
        < np.asarray(metrics["survivor_contact_time_s"])
    ):
        raise ValueError("evacuation precedes local contact")
    if metrics["firmware_robots"] and (
        not report["live_firmware_routing"]
        or metrics["firmware_requests"] < metrics["firmware_robots"]
    ):
        raise ValueError("selected firmware did not perform live route solves")


def check_source_archive(path, report):
    source = path.with_suffix(".source.tar.gz")
    with tarfile.open(source) as archive:
        metadata = json.load(archive.extractfile("bundle.json"))
        provenance = report["provenance"]
        if provenance["source_kind"] == "installed_distribution":
            if metadata.get("source_fingerprint") != provenance["source_fingerprint"]:
                raise ValueError("archive and installed source fingerprint differ")
            for item in metadata["source_files"]:
                payload = archive.extractfile("installed/" + item["path"]).read()
                if hashlib.sha256(payload).hexdigest() != item["sha256"]:
                    raise ValueError("archived source hash differs")
        elif (metadata["git_sha"], metadata["dirty_diff_hash"]) != (
            provenance["git_sha"],
            provenance["dirty_diff_hash"],
        ):
            raise ValueError("archive and Git source identity differ")
    return hashlib.sha256(source.read_bytes()).hexdigest()


def summarize(paths):
    rows, identities = [], set()
    for path in map(Path, paths):
        report = json.loads(path.read_text())
        issues = []
        try:
            check_rescue_report(report)
        except (ValueError, KeyError, TypeError) as error:
            issues.append(str(error))
        try:
            archive_sha = check_source_archive(path, report)
        except (ValueError, KeyError, OSError, tarfile.TarError) as error:
            issues.append(str(error))
            archive_sha = None
        metrics = report["metrics"]
        provenance = report["provenance"]
        fingerprint = provenance.get("source_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) not in (40, 64):
            issues.append("runtime source fingerprint is missing")
        else:
            identities.add(fingerprint)
        rows.append(
            {
                "seed": metrics["seed"],
                "robots": metrics["n_robots"],
                "accepted": not issues,
                "issues": issues,
                "survivors_evacuated": metrics["survivors_evacuated"],
                "last_evacuation_s": max(
                    (
                        value
                        for value in metrics["survivor_evacuation_time_s"]
                        if value is not None
                    ),
                    default=None,
                ),
                "survivor_exposure_proxy": sum(metrics["survivor_exposure_proxy"]),
                "firmware_requests": metrics["firmware_requests"],
                "report": str(path.resolve()),
                "source_bundle_sha256": archive_sha,
            }
        )
    return {
        "schema_version": 1,
        "contract": "physical-rescue-software-v1",
        "accepted": bool(rows)
        and all(row["accepted"] for row in rows)
        and len(identities) == 1,
        "runs": rows,
        "source_fingerprints": sorted(identities),
        "same_runtime_sources": len(identities) == 1,
        "physiology_calibrated": False,
        "hardware_cycles_measured": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path)
    parser.add_argument("--seeds", default="7,42,1")
    parser.add_argument("--robots", type=int, default=8)
    parser.add_argument("--firmware-robots", type=int, default=0)
    parser.add_argument("--firmware-image", type=Path)
    parser.add_argument("--tool-root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/rescue-validation"))
    args = parser.parse_args(argv)
    if args.reports:
        paths = args.reports
    else:
        seeds = parse_seed_spec(args.seeds)
        if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
            parser.error("seeds must be unique nonnegative integers")
        if not 3 <= args.robots <= 24 or not 0 <= args.firmware_robots <= 2:
            parser.error("robot count must be 3..24 and firmware workers 0..2")
        if args.firmware_robots and args.firmware_image is None:
            parser.error("firmware workers require --firmware-image")
        args.out.mkdir(parents=True, exist_ok=True)
        configuration = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        before = capture_manifest(0, configuration, {})
        (args.out / "protocol.json").write_text(before.to_json())
        paths = []
        for seed in seeds:
            path = args.out / f"seed{seed}.json"
            command = [
                "--seed",
                str(seed),
                "--robots",
                str(args.robots),
                "--firmware-robots",
                str(args.firmware_robots),
                "--out",
                str(args.out / "missions"),
                "--report",
                str(path),
            ]
            if args.firmware_image is not None:
                command += ["--firmware-image", str(args.firmware_image)]
            if args.tool_root is not None:
                command += ["--firmware-tool-root", str(args.tool_root)]
            with (args.out / f"seed{seed}.log").open("w") as log, redirect_stdout(log):
                rescue_main(command)
            paths.append(path)
        if (
            capture_manifest(0, configuration, {}).source_identity
            != before.source_identity
        ):
            raise RuntimeError("runtime sources changed during rescue validation")
    result = summarize(paths)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
