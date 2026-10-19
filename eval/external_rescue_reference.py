"""Reproduce a pinned, separately downloaded original-author rescue reference.

The external code is not vendored: its public repository declares no license.
This harness runs its untouched default entrypoint, then invokes the original
forward/reverse/brute-force allocation classes on the author's supplied table.
Optional fresh policy computation uses the author's supplied hazard episodes.
These are target-visiting/exit results, not human-escort mission results.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import runpy
import subprocess
import sys
import time
from pathlib import Path

REPOSITORY = "https://github.com/TihanyiD/multi_alloc"
COMMIT = "ddce51c4dc8c067d350ee89bbf41cea76b401f47"
CASE_DIRECTORY = Path("two_stage_framework/two_stage_framework_case_studies")


def _git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def _manifest(source):
    names = _git(source, "ls-files").splitlines()
    return {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names
    }


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _solution(solution):
    frame = solution.allocation.get_printable()
    result = {
        "algorithm": solution.algorithm,
        "objective": _json_value(solution.objective_value),
        "allocation": {
            str(robot): [str(task) for task in frame.columns if bool(row[task])]
            for robot, row in frame.iterrows()
        },
        "author_accounting_time_data_not_measured_wall_seconds": _json_value(
            solution.time_data
        ),
    }
    for name in ("alpha_G", "gamma_G"):
        if hasattr(solution, name):
            result[name] = float(getattr(solution, name))
    return result


def _worker(args):
    import matplotlib
    import numpy as np
    import pandas as pd

    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    harness = Path(__file__).resolve()
    harness_sha256 = hashlib.sha256(harness.read_bytes()).hexdigest()
    if _git(source, "rev-parse", "HEAD") != COMMIT:
        raise ValueError("external checkout is not the audited pinned commit")
    if _git(source, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("external tracked source or data has modifications")
    before = _manifest(source)
    directory = source / CASE_DIRECTORY
    sys.path.insert(0, str(directory))
    os.chdir(directory)
    started = time.perf_counter()
    with (
        (output / "original-entrypoint.stdout.txt").open("w") as stdout,
        contextlib.redirect_stdout(stdout),
    ):
        state = runpy.run_path(
            str(directory / "two_stage_framework.py"), run_name="__main__"
        )
    original_wall = time.perf_counter() - started
    original = {
        name: _solution(state[key])
        for name, key in (
            ("forward", "fg_solution"),
            ("reverse", "rg_solution"),
            ("brute_force_best", "bf_solution"),
            ("brute_force_worst", "worst_solution"),
        )
    }
    frame = state["function_frame"]
    parameters = state["parameters"]
    fresh = {}
    classes = (
        ("forward", state["Forward_Greedy_Allocator"]),
        ("reverse", state["Reverse_Greedy_Allocator"]),
        ("brute_force", state["Brute_Force_Allocator"]),
    )
    with (
        (output / "fresh-solvers.stdout.txt").open("w") as stdout,
        contextlib.redirect_stdout(stdout),
    ):
        for name, allocator_type in classes:
            started = time.perf_counter()
            allocator = allocator_type(frame)
            solution = allocator.solve_problem()
            if name == "brute_force":
                solution = solution[0]
            allocation_wall = time.perf_counter() - started
            policy_wall = None
            if args.fresh_policies and name != "brute_force":
                started = time.perf_counter()
                # Original methods; omit exhaustive curvature diagnostics,
                # which do not affect allocation or robot policies.
                allocator.add_optimal_policies(solution)
                allocator.add_optimal_paths(solution)
                allocator.add_group_objective(solution)
                policy_wall = time.perf_counter() - started
            payload = _solution(solution)
            expected = original["brute_force_best" if name == "brute_force" else name]
            actual_objective = solution.objective_value
            if isinstance(actual_objective, dict):
                actual_objective = actual_objective["multiplicative"]
            payload.update(
                measured_allocation_wall_seconds=allocation_wall,
                measured_policy_and_sample_replay_wall_seconds=policy_wall,
                same_allocation_as_supplied_result=payload["allocation"]
                == expected["allocation"],
                multiplicative_abs_error_vs_supplied_result=abs(
                    float(actual_objective) - expected["objective"]["multiplicative"]
                ),
            )
            fresh[name] = payload

    after = _manifest(source)
    if before != after:
        raise AssertionError(
            "running the external example changed tracked source or data"
        )
    if hashlib.sha256(harness.read_bytes()).hexdigest() != harness_sha256:
        raise AssertionError("the local reproduction harness changed during execution")
    # The dynamic program's supplied objective table is exported as data for
    # transparent auditing. It is not recast as an escort-rescue objective.
    values = np.asarray(frame["value"].matrix)
    table = {
        "robot_ids": [str(robot.id) for robot in frame.robots],
        "task_subsets": [
            sorted(map(str, subset)) for subset in frame["value"].domain[1]
        ],
        "values": values.tolist(),
        "provenance": "author-supplied cached dynamic-program objective table",
    }
    (output / "author-objective-table.json").write_text(
        json.dumps(table, indent=2) + "\n"
    )
    receipt = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "commit": COMMIT,
        "commit_url": f"{REPOSITORY}/commit/{COMMIT}",
        "harness": {
            "path": str(harness),
            "sha256": harness_sha256,
            "unchanged_during_execution": True,
        },
        "license": (
            "No LICENSE or license declaration found in the original repository; "
            "source was downloaded separately, not vendored."
        ),
        "external_source_path": str(source),
        "source_manifest_sha256": before,
        "tracked_source_and_supplied_data_unchanged": before == after,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
            "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "problem": {
            "case": "case_study_1",
            "map_shape": list(parameters.map.shape),
            "robots": len(parameters.robots),
            "targets": len(parameters.tasks),
            "hazard_samples": int(parameters.E),
            "time_states": int(parameters.N),
            "objective_table_shape": list(values.shape),
        },
        "original_entrypoint": {
            "source_modified": False,
            "measured_wall_seconds": original_wall,
            "result_kind": (
                "unchanged author entrypoint replaying supplied samples, "
                "objective table and saved solutions"
            ),
            "results": original,
        },
        "fresh_original_solvers": {
            "result_kind": (
                "original author allocation methods executed afresh on "
                "author-supplied cached DP objective table"
            ),
            "fresh_low_level_policies": args.fresh_policies,
            "curvature_diagnostics_recomputed": False,
            "hazard_samples_regenerated": False,
            "full_objective_table_regenerated": False,
            "results": fresh,
        },
        "limits": [
            (
                "Targets are visited and robots exit; no survivor waiting, pickup, "
                "human escort, radio blackout, or exposure-dose extension is included."
            ),
            (
                "Empirical group objectives use the author's same supplied 5000 "
                "hazard samples; they are replay values, not independent held-out "
                "estimates."
            ),
            (
                "Original source timing fields include supplied historic function "
                "evaluation times and must not be used as present measured runtime."
            ),
            (
                "No comparison against the proposed common-source certificate or "
                "the production MuJoCo controller is established by this receipt."
            ),
        ],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {"receipt": str(output / "receipt.json"), "fresh_original_solvers": fresh},
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--fresh-policies", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.python.absolute()),
        str(Path(__file__).resolve()),
        "--worker",
        "--source",
        str(args.source.resolve()),
        "--output",
        str(args.output.resolve()),
    ]
    if args.fresh_policies:
        command.append("--fresh-policies")
    environment = dict(os.environ)
    environment.update(
        MPLBACKEND="Agg",
        MPLCONFIGDIR="/tmp/swarmrescue-tihanyi-mpl",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        PYTHONHASHSEED="0",
    )
    subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
