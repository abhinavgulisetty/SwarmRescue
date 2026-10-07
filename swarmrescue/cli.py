"""Installed entry point for mission, evaluation and firmware commands."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

COMMANDS = {
    "mission": "cascade.demo",
    "rescue": "swarmrescue.rescue",
    "cases": "swarmrescue.cases",
    "rescue-evaluation": "eval.rescue",
    "tactical": "ember.demo",
    "view": "cascade.view",
    "evaluate": "eval.procedural",
    "solver-benchmark": "eval.solver_benchmark",
    "range-evaluation": "eval.range_delivery",
    "firmware-replay": "eval.firmware_replay",
    "firmware-build": "mcu.runtime",
    "deploy": "swarmrescue.deploy",
}


def doctor():
    checks = []

    def check(name, run):
        try:
            detail = run()
            checks.append({"name": name, "passed": True, "detail": detail})
        except Exception as error:
            checks.append({"name": name, "passed": False, "detail": str(error)})

    def native_route():
        import numpy as np

        from ember.routing import LMDPConfig, LMDPModel
        from mcu.native_log_lmdp import solve_native_log_q16

        model = LMDPModel(
            np.arange(4),
            np.array([1, 2, 2]),
            np.ones(3),
            np.ones(3),
            np.array([0, 0, 1], bool),
            np.zeros(3, bool),
            LMDPConfig(),
        )
        result = solve_native_log_q16(model, strategy="auto")
        if not result.fixed_point or result.route(0) != (0, 1, 2):
            raise RuntimeError("native integer smoke route failed")
        return "C kernel compiled and returned the expected route"

    def physics():
        import mujoco

        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        data = mujoco.MjData(model)
        mujoco.mj_step(model, data)
        return version("mujoco")

    def assets():
        import firmware

        root = Path(firmware.__file__).parent
        required = (
            "log_lmdp.c",
            "log_lmdp.h",
            "log_lmdp_lut.h",
            "deadline.c",
            "deadline.h",
            "replay/replay.c",
            "replay/mps2.ld",
            "runtime/service.c",
            "runtime/host.c",
            "runtime/mps2.c",
            "runtime/mps2.ld",
        )
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise RuntimeError("missing firmware assets: " + ", ".join(missing))
        return {"root": str(root), "files": len(required)}

    def interpreter():
        if not (3, 11) <= sys.version_info[:2] < (3, 13):
            raise RuntimeError("supported Python versions are 3.11 and 3.12")
        return sys.version.split()[0]

    check("python", interpreter)
    check("firmware-assets", assets)
    check("native-routing", native_route)
    check("mujoco", physics)
    try:
        package_version = version("swarmrescue")
    except PackageNotFoundError:
        package_version = "source checkout"
    return {
        "package_version": package_version,
        "ready": all(c["passed"] for c in checks),
        "checks": checks,
        "supported_commands": list(COMMANDS),
        "optional_tools": {
            "qemu": shutil.which("qemu-system-arm"),
            "arm_gcc": shutil.which("arm-none-eabi-gcc"),
        },
        "whole_project_acceptance_gate_met": False,
    }


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="swarmrescue", description=__doc__)
    parser.add_argument("command", choices=["doctor", *COMMANDS])
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    if args.command == "doctor":
        if args.arguments:
            parser.error("doctor takes no additional arguments")
        result = doctor()
        print(json.dumps(result, indent=2))
        return 0 if result["ready"] else 1
    return importlib.import_module(COMMANDS[args.command]).main(args.arguments)


if __name__ == "__main__":
    raise SystemExit(main())
