"""Install and verify a local simulator/firmware deployment from a release wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def install(wheel, prefix, *, tool_root=None):
    wheel = Path(wheel).resolve()
    prefix = Path(prefix).resolve()
    if not wheel.is_file():
        raise ValueError("release wheel is missing")
    if Path(sys.prefix).resolve().is_relative_to(prefix):
        raise ValueError("run the installer from outside the target deployment")
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for the locked local installation")
    with zipfile.ZipFile(wheel) as archive:
        lock = archive.read("swarmrescue/runtime-lock.txt")
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / "runtime-lock.txt").write_bytes(lock)
    environment = prefix / "venv"
    python = environment / "bin/python"
    if not python.exists():
        subprocess.run(
            [uv, "venv", "--python", sys.executable, str(environment)], check=True
        )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--require-hashes",
            "-r",
            str(prefix / "runtime-lock.txt"),
        ],
        check=True,
    )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "--reinstall",
            str(wheel),
        ],
        check=True,
    )
    cli = environment / "bin/swarmrescue"
    # An empty working directory verifies that installed imports/assets stand alone.
    with tempfile.TemporaryDirectory(prefix="swarmrescue-deployment-") as folder:
        doctor = subprocess.run(
            [str(cli), "doctor"], cwd=folder, check=True, capture_output=True, text=True
        )
        health = json.loads(doctor.stdout)
        provenance = subprocess.run(
            [
                str(python),
                "-c",
                "import json; from dataclasses import asdict; "
                "from ember.manifest import capture_manifest; "
                "print(json.dumps(asdict(capture_manifest(0,{},{}))))",
            ],
            cwd=folder,
            check=True,
            capture_output=True,
            text=True,
        )
        identity = json.loads(provenance.stdout)
        if identity["source_kind"] != "installed_distribution":
            raise RuntimeError(
                "deployment imported source checkout instead of the wheel"
            )
        firmware = None
        if tool_root is not None:
            tool_root = Path(tool_root).resolve()
            command = [
                str(cli),
                "firmware-build",
                "--arm",
                "--tool-root",
                str(tool_root),
                "--out",
                str(prefix / "firmware"),
            ]
            build = subprocess.run(
                command, cwd=folder, check=True, capture_output=True, text=True
            )
            firmware = json.loads(build.stdout)
            probe = subprocess.run(
                [
                    str(python),
                    "-c",
                    "import json, sys; from mcu.runtime import FirmwareBackend\n"
                    "with FirmwareBackend(sys.argv[1], arm=True, "
                    "tool_root=sys.argv[2], timeout_s=10) as backend:\n"
                    "    print(json.dumps({'passed': True, "
                    "'emulator': backend.metadata['emulator'], "
                    "'image_sha256': backend.image_sha256}))\n",
                    firmware["executable"],
                    str(tool_root),
                ],
                cwd=folder,
                check=True,
                capture_output=True,
                text=True,
            )
            firmware["startup_check"] = json.loads(probe.stdout)
    (prefix / "doctor.json").write_text(json.dumps(health, indent=2) + "\n")
    launcher = write_launcher(prefix, cli, firmware=firmware, tool_root=tool_root)
    report = {
        "schema_version": 1,
        "target": "local-simulator-and-cortex-m4" if firmware else "local-simulator",
        "ready": health["ready"],
        "package_version": health["package_version"],
        "wheel": str(wheel),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "installed_source_fingerprint": identity["source_fingerprint"],
        "runtime_lock_sha256": hashlib.sha256(lock).hexdigest(),
        "prefix": str(prefix),
        "launcher": str(launcher),
        "firmware": firmware,
        "whole_robot_firmware": False,
        "physical_board_flashed": False,
        "whole_project_acceptance_gate_met": False,
    }
    (prefix / "deployment.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def write_launcher(prefix, cli, *, firmware=None, tool_root=None):
    """Write a standalone POSIX launcher with literal paths and forwarded options."""
    command = [
        str(cli),
        "mission",
        "--seed",
        "42",
        "--robots",
        "8",
        "--compact",
        "--live-disaster",
        "--distributed",
        "--region-cells",
        "4",
        "--latest-evidence",
        "--range-gbp",
        "--range-radio",
        "--no-render",
        "--out",
        str(prefix / "missions"),
        "--report",
        str(prefix / "latest-mission.json"),
    ]
    if firmware:
        command += [
            "--firmware-robots",
            "2",
            "--firmware-image",
            firmware["executable"],
            "--firmware-tool-root",
            str(tool_root),
        ]
    launcher = prefix / "run-mission"
    launcher.write_text(
        "#!/bin/sh\nset -eu\ncd "
        + shlex.quote(str(prefix))
        + "\n"
        + "export SWARMRESCUE_CACHE_DIR="
        + shlex.quote(str(prefix / "cache"))
        + "\n"
        + "exec "
        + shlex.join(command)
        + ' "$@"\n'
    )
    launcher.chmod(0o755)
    return launcher


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, default=Path("outputs/deployment"))
    parser.add_argument("--tool-root", type=Path)
    args = parser.parse_args(argv)
    report = install(args.wheel, args.prefix, tool_root=args.tool_root)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
