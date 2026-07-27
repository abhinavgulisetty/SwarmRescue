"""Phase 0 demo: generate a collapsed building and drive a robot through it.

    python -m cascade.demo --seed 7

Generates a seeded building, collapses it by simulation, extracts the
traversability graph from the *settled* rubble, plans a route from a
ground-floor exit to an upper storey, and drives a teleoperated robot along
it -- reporting which storeys the robot's own trajectory actually reached
rather than which ones the planner believed were reachable.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .collapse import apply_settled_state, collapse_for_seed, model_hash
from .extract import (
    build_graph,
    nearest_node,
    rasterize,
    reachable_storeys,
    shortest_path,
)
from .floorplan import build_floorplan
from .mjcf import settled_poses, to_mjcf, to_settled_mjcf
from .params import BuildingParams, CascadeParams
from .render import CameraView, render_scene, save_png
from .robot import RobotSpec, WaypointDriver, build_robot, place_robot
from .structure import build_structure

#: Physics steps between rendered frames.
FRAME_EVERY: int = 25
#: Give up if the robot has not reached the goal in this many steps.
MAX_DRIVE_STEPS: int = 60_000
#: How far outside its exit the robot starts, metres.
SPAWN_STANDOFF_M: float = 1.2


@dataclass(frozen=True, slots=True)
class DemoResult:
    seed: int
    storeys_reached: set[int]
    target_storey: int
    path_nodes: int
    planner_reachable: set[int]
    steps: int
    frames: int
    max_height_m: float

    @property
    def succeeded(self) -> bool:
        return self.target_storey in self.storeys_reached

    def report(self) -> str:
        status = "REACHED" if self.succeeded else "DID NOT REACH"
        return (
            f"seed {self.seed}: {status} storey {self.target_storey}\n"
            f"  planner says reachable : {sorted(self.planner_reachable)}\n"
            f"  robot actually visited : {sorted(self.storeys_reached)}\n"
            f"  route                  : {self.path_nodes} graph nodes\n"
            f"  simulated              : {self.steps} steps, "
            f"max height {self.max_height_m:.2f} m"
        )


def _spawn_pose(plan, standoff: float) -> tuple[tuple[float, float, float], float]:
    """Where the robot starts: just outside an exit, facing in."""
    door = plan.exits[0]
    footprint = plan.footprint
    x, y = door.center
    if door.axis.value == "x":
        outward = -1.0 if abs(door.at - footprint.x0) < 1e-6 else 1.0
        return (x + outward * standoff, y, 0.12), (math.pi if outward > 0 else 0.0)
    outward = -1.0 if abs(door.at - footprint.y0) < 1e-6 else 1.0
    return (x, y + outward * standoff, 0.12), (
        -0.5 * math.pi if outward > 0 else 0.5 * math.pi
    )


def run_demo(
    seed: int,
    target_storey: int,
    params: CascadeParams,
    out_dir: Path,
    *,
    render: bool = True,
    use_cache: bool = True,
) -> DemoResult:
    building: BuildingParams = params.building
    plan = build_floorplan(seed, building)
    structure = build_structure(plan)

    spawn, yaw = _spawn_pose(plan, SPAWN_STANDOFF_M)
    spec = RobotSpec()

    # Phase 1: collapse the building on its own.  The robot is not in this
    # model at all, so the cached rubble is a property of the seed rather than
    # of whatever platform happens to be driving through it.
    dynamic_xml = to_mjcf(structure, plan)
    digest = model_hash(dynamic_xml)
    dynamic = mujoco.MjModel.from_xml_string(dynamic_xml)
    print(f"[cascade] seed {seed}: {dynamic.nbody - 1} bodies, {dynamic.neq} welds")

    collapse = collapse_for_seed(
        dynamic, structure, seed, params.collapse, digest, use_cache=use_cache
    )
    print(f"[cascade] {collapse.summary}")
    settled = mujoco.MjData(dynamic)
    apply_settled_state(dynamic, settled, collapse)

    # Phase 2: bake the settled rubble as static geometry and add the robot.
    poses = settled_poses(dynamic, settled, structure)
    robot_body, actuators = build_robot(spec, spawn)
    model = mujoco.MjModel.from_xml_string(
        to_settled_mjcf(
            structure,
            plan,
            poses,
            extra_worldbody=[robot_body],
            extra_actuators=actuators,
        )
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    place_robot(model, data, spec, spawn, yaw)
    print(
        f"[cascade] settled scene: {len(poses)} members standing, "
        f"{model.nbody - 1} dynamic bodies"
    )

    grid = rasterize(
        model, data, building.footprint_m, building.total_height_m, params.traversability
    )
    graph = build_graph(grid, params.traversability)
    print(
        f"[cascade] traversability graph: {graph.n_nodes} cells, "
        f"{graph.n_edges} moves, occupancy {grid.occupancy_fraction():.3f}"
    )

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.name)
    start = nearest_node(graph, tuple(data.xpos[body_id]))
    planner_reachable = reachable_storeys(graph, start, building.storey_height_m)

    stairwell = plan.stairwells()[0].rect
    goal_point = (
        stairwell.center[0],
        stairwell.center[1],
        target_storey * building.storey_height_m + 0.1,
    )
    goal = nearest_node(graph, goal_point)
    route = shortest_path(graph, start, goal)
    if not route:
        print(
            f"[cascade] no route from the exit to storey {target_storey}; "
            f"planner reachable: {sorted(planner_reachable)}",
            file=sys.stderr,
        )

    waypoints = graph.world_positions()[route] if route else np.zeros((0, 3))
    driver = WaypointDriver(spec=spec, waypoints=waypoints)

    frames = 0
    visited: set[int] = set()
    max_height = 0.0
    step = 0
    renderer = (
        mujoco.Renderer(model, height=720, width=1120) if render else None
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        while step < MAX_DRIVE_STEPS and not driver.finished and not driver.stalled:
            driver.step(model, data)
            mujoco.mj_step(model, data)
            step += 1

            height = float(data.xpos[body_id][2])
            max_height = max(max_height, height)
            visited.add(int(math.floor(height / building.storey_height_m + 0.25)))

            if renderer is not None and step % FRAME_EVERY == 0:
                position = tuple(float(v) for v in data.xpos[body_id])
                view = CameraView("chase", position, 7.0, 130.0, -22.0)
                save_png(
                    out_dir / f"seed{seed}_frame{frames:04d}.png",
                    render_scene(model, data, view, renderer=renderer),
                )
                frames += 1
    finally:
        if renderer is not None:
            renderer.close()

    if driver.stalled:
        print(
            f"[cascade] robot stalled at waypoint {driver.index}/{len(waypoints)} "
            "-- geometry the planner believed was open",
            file=sys.stderr,
        )

    return DemoResult(
        seed=seed,
        storeys_reached={s for s in visited if s >= 0},
        target_storey=target_storey,
        path_nodes=len(route),
        planner_reachable=planner_reachable,
        steps=step,
        frames=frames,
        max_height_m=max_height,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cascade.demo", description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--storey",
        type=int,
        default=2,
        help="storey the robot should reach (0 is the ground floor)",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
    parser.add_argument(
        "--no-render", action="store_true", help="skip frame capture"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="re-simulate the collapse"
    )
    args = parser.parse_args(argv)

    result = run_demo(
        seed=args.seed,
        target_storey=args.storey,
        params=CascadeParams(),
        out_dir=args.out,
        render=not args.no_render,
        use_cache=not args.no_cache,
    )
    print(result.report())
    if result.frames:
        print(f"  frames written to {args.out}")
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
