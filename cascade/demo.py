"""Phase 0 demo: collapse a building and drive a swarm through the ruin.

    python -m cascade.demo --seed 7

Generates a seeded building, collapses it by simulation, extracts the
traversability graph from the *settled* rubble, spreads a swarm of robots over
well-separated goals, drives them, and writes an annotated video -- reporting
which storeys the robots' own trajectories reached rather than which ones the
planner believed were reachable.

The distinction between those two numbers is the entire point of running this
rather than trusting the graph.  A planner that certifies a route the robot
cannot drive is worse than no planner, and the first version of this demo did
exactly that: it planned for a point robot, and a 0.34 x 0.24 m chassis wedged
itself in a 0.25 m gap sixteen waypoints in.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .cinema import (
    CEILING_MARGIN_M,
    PLAN_FLOOR_DROP_M,
    GeomGroups,
    PathTrace,
    SwarmCamera,
    Trails,
    draw_paths,
    hero_style,
    index_geoms,
    plan_style,
    plan_view,
)
from .collapse import apply_settled_state, collapse_for_seed, model_hash
from .extract import build_graph, rasterize
from .floorplan import build_floorplan
from .hud import RobotReadout, VideoWriter, caption, draw_hud, stack_panes
from .mjcf import settled_poses, to_mjcf, to_settled_mjcf
from .params import BuildingParams, CascadeParams
from .render import render_scene
from .robot import RobotSpec
from .structure import build_structure
from .swarm import Swarm, plan_swarm, swarm_bodies

#: Give up if the swarm has not finished in this many steps.  The longest
#: routes run to the roof and are ~65 m; at the cornering-safe cruise speed of
#: 0.5 m/s that is around 130 s of simulated time, and the budget has to cover
#: that plus the staggered departures, the switchbacks a robot creeps through,
#: and a few recovery manoeuvres each.
MAX_DRIVE_STEPS: int = 160_000
#: Steps of gravity-only settling before the drivers are switched on, so a
#: robot dropped a few centimetres onto rubble lands before it tries to move.
SETTLE_STEPS: int = 250
#: Hard cap on rendered frames, so a run that never terminates cannot
#: produce an unwatchably long video.  At 24 fps this is two minutes.
MAX_FRAMES: int = 2880
#: Renderer geom budget.  The settled scene is ~460 static geoms; on top of
#: that go four planned routes and four trails, drawn as capsules.  The default
#: 10000 is ample, but the overlay fails *silently* when it runs out -- a trail
#: that stops halfway looks like a robot that stopped halfway -- so the budget
#: is stated rather than inherited.
MAX_SCENE_GEOMS: int = 12_000


@dataclass(frozen=True, slots=True)
class DemoResult:
    seed: int
    n_robots: int
    storeys_reached: set[int]
    planner_reachable: set[int]
    per_robot: tuple[tuple[str, int, int, str], ...]
    steps: int
    frames: int
    video: Path | None

    @property
    def arrived(self) -> int:
        return sum(1 for _, _, _, status in self.per_robot if status == "arrived")

    @property
    def succeeded(self) -> bool:
        """At least one robot reached an upper storey and none of them stalled."""
        upper = {s for s in self.storeys_reached if s >= 1}
        return bool(upper) and all(
            status != "stalled" for _, _, _, status in self.per_robot
        )

    def report(self) -> str:
        lines = [
            f"seed {self.seed}: {self.arrived}/{self.n_robots} robots arrived",
            f"  planner says reachable : {sorted(self.planner_reachable)}",
            f"  swarm actually visited : {sorted(self.storeys_reached)}",
            f"  simulated              : {self.steps} steps "
            f"({self.steps * 0.002:.1f} s)",
        ]
        for name, storey, goal, status in self.per_robot:
            lines.append(f"    {name:<8} storey {storey} -> {goal}   {status}")
        if self.video:
            lines.append(f"  video                  : {self.video}")
        return "\n".join(lines)


def _settled_scene(seed, params, plan, structure, use_cache):
    """Collapse the building and return its settled member poses."""
    dynamic_xml = to_mjcf(structure)
    digest = model_hash(dynamic_xml)
    dynamic = mujoco.MjModel.from_xml_string(dynamic_xml)
    print(f"[cascade] seed {seed}: {dynamic.nbody - 1} bodies, {dynamic.neq} welds")

    collapse = collapse_for_seed(
        dynamic, structure, seed, params.collapse, digest, use_cache=use_cache
    )
    print(f"[cascade] {collapse.summary}")
    settled = mujoco.MjData(dynamic)
    apply_settled_state(dynamic, settled, collapse)
    return settled_poses(dynamic, settled, structure)


def run_demo(
    seed: int,
    params: CascadeParams,
    out_dir: Path,
    *,
    n_robots: int = 4,
    render: bool = True,
    use_cache: bool = True,
    fps: int = 30,
    speed: float = 2.0,
    width: int = 1120,
    height: int = 700,
) -> DemoResult:
    building = params.building
    plan = build_floorplan(seed, building)
    structure = build_structure(plan)
    poses = _settled_scene(seed, params, plan, structure, use_cache)

    # The graph is extracted from a scene with *no robots in it*.  Rasterizing
    # a model that already contains them marks each robot's own chassis as an
    # obstacle, which -- once obstacles are grown by the robot radius -- can
    # block the very cell it is standing on.
    bare = mujoco.MjModel.from_xml_string(to_settled_mjcf(structure, poses))
    bare_data = mujoco.MjData(bare)
    mujoco.mj_forward(bare, bare_data)

    grid = rasterize(
        bare,
        bare_data,
        building.footprint_m,
        building.total_height_m,
        params.traversability,
    )
    graph = build_graph(grid, params.traversability, bare, bare_data)
    print(
        f"[cascade] traversability graph: {graph.n_nodes} cells, "
        f"{graph.n_edges} moves, occupancy {grid.occupancy_fraction():.3f}"
    )

    assignments = plan_swarm(plan, graph, building, n_robots, RobotSpec())
    routed = [a for a in assignments if len(a.route)]
    if not routed:
        print("[cascade] no robot could be routed anywhere", file=sys.stderr)
    planner_reachable = {a.goal_storey for a in routed}
    for item in assignments:
        print(
            f"[cascade]   {item.name}: {len(item.route)} waypoints "
            f"-> storey {item.goal_storey}"
        )

    bodies, actuators = swarm_bodies(assignments)
    model = mujoco.MjModel.from_xml_string(
        to_settled_mjcf(
            structure,
            poses,
            extra_worldbody=bodies,
            extra_actuators=actuators,
        )
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    swarm = Swarm(assignments)
    swarm.place(model, data)
    print(
        f"[cascade] settled scene: {len(poses)} members standing, "
        f"{len(swarm.members)} robots, {model.nbody - 1} dynamic bodies"
    )

    # Let the robots drop onto the ground before they are asked to drive.
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(model, data)
    _check_stable(model, data)

    stage = Stage(
        seed=seed,
        building=building,
        groups=index_geoms(model, data),
        camera=SwarmCamera(footprint_m=building.footprint_m),
        trails=Trails(n_robots=len(swarm.members)),
        aspect=width / height,
    )
    every = max(1, round(speed / (fps * params.collapse.timestep_s)))

    renderer = (
        mujoco.Renderer(model, height=height, width=width, max_geom=MAX_SCENE_GEOMS)
        if render
        else None
    )
    writer = (
        VideoWriter(out_dir / f"seed{seed}_swarm.mp4", fps=fps) if render else None
    )
    step = 0

    try:
        while step < MAX_DRIVE_STEPS and not swarm.done:
            swarm.step(model, data)
            mujoco.mj_step(model, data)
            step += 1
            swarm.observe(data, building.storey_height_m)

            if renderer is None or writer is None:
                continue
            if step % every or writer.frames >= MAX_FRAMES:
                continue
            writer.append(
                stage.compose(
                    model,
                    data,
                    renderer,
                    swarm,
                    elapsed_s=step * params.collapse.timestep_s,
                )
            )
    finally:
        if renderer is not None:
            renderer.close()

    video = None
    frames = 0
    if writer is not None:
        frames = writer.frames
        video = writer.close()
        print(f"[cascade] wrote {frames} frames -> {video}")

    per_robot = tuple(
        (
            m.name,
            max(m.visited) if m.visited else 0,
            m.assignment.goal_storey,
            m.status,
        )
        for m in swarm.members
    )
    return DemoResult(
        seed=seed,
        n_robots=len(swarm.members),
        storeys_reached=swarm.storeys_reached,
        planner_reachable=planner_reachable,
        per_robot=per_robot,
        steps=step,
        frames=frames,
        video=video,
    )


def _check_stable(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Report a diverged solver rather than driving a broken simulation."""
    bad = int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    if bad:
        print(
            f"[cascade] WARNING: solver reported {bad} bad-acceleration events "
            "during settling; a robot may have spawned inside geometry",
            file=sys.stderr,
        )
    if not np.all(np.isfinite(data.qpos)):
        raise RuntimeError("simulation diverged during settling")


@dataclass
class Stage:
    """Everything the presentation needs that the simulation does not.

    A class rather than a pile of parameters because two of these -- the camera
    and the trails -- carry state between frames, and threading them through a
    free function alongside eight other arguments was how `_compose` came to
    take ten.
    """

    seed: int
    building: BuildingParams
    groups: GeomGroups
    camera: SwarmCamera
    trails: Trails
    aspect: float

    def compose(
        self,
        model,
        data,
        renderer,
        swarm: Swarm,
        *,
        elapsed_s: float,
    ) -> np.ndarray:
        """Render both panes and wrap them in the telemetry overlay."""
        positions = swarm.positions(data)
        self.trails.record(positions)
        subject = swarm.focus()
        subject_at = np.asarray(data.xpos[subject.body_id], dtype=np.float64)
        ceiling = self.camera.ceiling(subject_at)
        traces = self._traces(swarm)

        hero_style(model, self.groups, ceiling)
        hero = caption(
            render_scene(
                model,
                data,
                self.camera.update(subject_at, positions),
                renderer=renderer,
                overlay=lambda scene: draw_paths(scene, traces),
            ),
            f"tracking {subject.name}  ·  cut {CEILING_MARGIN_M:.1f} m above it",
        )

        floor_z = float(subject_at[2]) - PLAN_FLOOR_DROP_M
        plan_style(model, self.groups, floor_z, ceiling)
        top = caption(
            render_scene(
                model,
                data,
                plan_view(self.building.footprint_m, floor_z, aspect=self.aspect),
                renderer=renderer,
                overlay=lambda scene: draw_paths(scene, traces),
            ),
            f"plan of storey {self._storey(subject_at[2])}",
        )

        return draw_hud(
            stack_panes(hero, top),
            title="EMBER · CASCADE",
            subtitle=(
                f"Phase 0 — swarm traversal of a blast-collapsed "
                f"{self.building.n_storeys}-storey building  ·  seed {self.seed}"
            ),
            readouts=self._readouts(data, swarm, subject),
            elapsed_s=elapsed_s,
            n_storeys=self.building.n_storeys + 1,
        )

    def _storey(self, height_m: float) -> int:
        return math.floor(height_m / self.building.storey_height_m + 0.25)

    def _traces(self, swarm: Swarm) -> list[PathTrace]:
        return [
            PathTrace(
                livery=member.assignment.spec.livery,
                route=member.assignment.route,
                trail=self.trails.of(index),
            )
            for index, member in enumerate(swarm.members)
        ]

    def _readouts(self, data, swarm: Swarm, subject) -> list[RobotReadout]:
        return [
            RobotReadout(
                name=member.name,
                livery=member.assignment.spec.livery,
                storey=self._storey(float(data.xpos[member.body_id][2])),
                goal_storey=member.assignment.goal_storey,
                progress=member.progress,
                status=member.status,
                is_subject=member is subject,
            )
            for member in swarm.members
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cascade.demo", description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--robots", type=int, default=4, help="how many robots to deploy"
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--speed",
        type=float,
        default=2.0,
        help="playback speed relative to real time",
    )
    parser.add_argument("--width", type=int, default=1120)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--no-render", action="store_true", help="skip video")
    parser.add_argument(
        "--no-cache", action="store_true", help="re-simulate the collapse"
    )
    args = parser.parse_args(argv)

    result = run_demo(
        seed=args.seed,
        params=CascadeParams(),
        out_dir=args.out,
        n_robots=args.robots,
        render=not args.no_render,
        use_cache=not args.no_cache,
        fps=args.fps,
        speed=args.speed,
        width=args.width,
        height=args.height,
    )
    print(result.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
