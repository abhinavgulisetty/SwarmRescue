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
import json
import math
import sys
from dataclasses import asdict, dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from ember.localization import RSSIConfig, rssi_to_range
from ember.manifest import capture_manifest, write_source_bundle
from ember.physical import PhysicalRouteResult, plan_belief_route
from ember.physical_team import (
    PhysicalRobotTeam,
    collision_aware_params,
    ground_mission_graph,
    place_collision_free_team,
    rebuild_physical_graph,
)
from ember.search import (
    CooperativeSearchResult,
    localize_and_assign,
    sample_survivor_sites,
)
from hazard import HazardDomain, HazardSolver

from .cinema import (
    CEILING_MARGIN_M,
    PLAN_FLOOR_DROP_M,
    GeomGroups,
    PathTrace,
    SwarmCamera,
    Trails,
    draw_paths,
    event_style,
    hero_style,
    highlight_members,
    index_geoms,
    plan_style,
    plan_view,
)
from .collapse import (
    CollapseResult,
    apply_settled_state,
    collapse_for_seed,
    model_hash,
)
from .damage import compute_damage
from .extract import (
    GRID_MARGIN_M,
    TraversabilityGraph,
    build_graph,
    dijkstra_from,
    nearest_node,
    path_from,
    rasterize,
)
from .floorplan import build_floorplan
from .hud import (
    ResearchMetric,
    RobotReadout,
    VideoWriter,
    draw_hud,
    research_card,
    stack_panes,
)
from .mjcf import settled_poses, to_mjcf, to_settled_mjcf
from .params import BuildingParams, CascadeParams, compact_demo_params
from .render import CameraView, render_scene
from .rng import stage_rng
from .robot import RobotSpec, RobotState
from .structure import MemberKind, Structure, build_structure
from .swarm import Assignment, Swarm, plan_swarm, swarm_bodies
from .thermal import ThermalDamage, ThermalDamageConfig

#: Give up if the swarm has not finished in this many steps.  The longest
#: routes run to the roof and are ~65 m; at the cornering-safe cruise speed of
#: 0.5 m/s that is around 130 s of simulated time, and the budget has to cover
#: that plus the staggered departures, the switchbacks a robot creeps through,
#: and a few recovery manoeuvres each.
MAX_DRIVE_STEPS: int = 160_000
#: Hard cap on rendered frames, so a run that never terminates cannot
#: produce an unwatchably long video.  At 24 fps this is two minutes.
MAX_FRAMES: int = 2880
#: Renderer geom budget.  The settled scene is ~460 static geoms; on top of
#: that go four planned routes and four trails, drawn as capsules.  The default
#: 10000 is ample, but the overlay fails *silently* when it runs out -- a trail
#: that stops halfway looks like a robot that stopped halfway -- so the budget
#: is stated rather than inherited.
MAX_SCENE_GEOMS: int = 12_000
#: Hold the truthful terminal state long enough to be read in the encoded demo.
FINAL_HOLD_S: float = 4.0
#: Minimum time the hero camera stays with one robot before progress rankings
#: may cut to another.  Event states can still pre-empt the shot immediately.
SHOT_MIN_S: float = 5.0
#: Let a failure remain on screen briefly before following another robot.
SHOT_TERMINAL_GRACE_S: float = 1.5
#: Bounded dynamic mission: long enough to establish destinations, enter the
#: building, reroute, and then react to a secondary collapse.
LIVE_DISASTER_SECONDS: float = 48.0
#: Damage accumulation starts only after this point.  Previously it began at
#: frame zero and failed the facade while robots were still entering, so the
#: clip showed an entrance pile-up instead of a response to a new obstacle.
LIVE_DISASTER_AT_S: float = 12.0
#: Robots deploy first, then fuse their BLE observations and choose targets at
#: runtime. Ground-truth survivor positions never enter the routing call.
LIVE_SEARCH_LOCALIZE_AT_S: float = 7.5
LIVE_SEARCH_MIN_ROUTE_COST: float = 24.0
LIVE_SEARCH_MAX_ROUTE_COST: float = 48.0
LIVE_SEARCH_MAX_ERROR_M: float = 2.0
#: Let all four robots visibly deploy before thermal failure in the bounded clip.
LIVE_DISASTER_STAGGER_STEPS: int = 50
#: Ground-floor compact search can safely use the robot's 0.45 m/s design
#: cruise. Full multi-storey traversal retains its slower stair-safe default.
LIVE_DISASTER_CRUISE_RAD_S: float = 6.0
#: More than one chassis length: visible motion, not settling jitter.
LIVE_DISASTER_MIN_TRAVEL_M: float = 0.35
#: Twenty-eight graph cells put the goals several metres indoors, after the
#: exterior staging points, and beyond a small, locally avoidable fire.
LIVE_DISASTER_ROUTE_WAYPOINTS: int = 28
LIVE_DISASTER_MIN_INTERIOR_DEPTH_M: float = 1.5
#: Demo calibration: roughly 1.5 s at the 1100 K ignition temperature.
LIVE_THERMAL_EXPOSURE_K_S: float = 800.0
LIVE_HAZARD_DT_S: float = 0.1
LIVE_ROUTING_HAZARD_WAYPOINT: int = 10
LIVE_ROUTING_HAZARD_RADIUS_M: float = 1.0
LIVE_ROUTING_SCAN_RADIUS_M: float = 6.0
LIVE_COLLAPSE_SCAN_RADIUS_M: float = 10.0
LIVE_ROUTING_HAZARD_HEIGHT_M: float = 1.4
#: A compact structure needs a proportionally broader heated patch for a
#: secondary failure to read as a bay event rather than two isolated members.
COMPACT_FOOTPRINT_AREA_M2: float = 200.0
FULL_THERMAL_RADIUS_FACTOR: float = 0.70
COMPACT_THERMAL_RADIUS_FACTOR: float = 1.20


def _thermal_damage_active(step: int, timestep_s: float) -> bool:
    """Whether the staged secondary collapse may start accumulating damage."""

    return step * timestep_s >= LIVE_DISASTER_AT_S


def _thermal_damage_radius(params: CascadeParams) -> float:
    area = params.building.footprint_m[0] * params.building.footprint_m[1]
    factor = (
        COMPACT_THERMAL_RADIUS_FACTOR
        if area <= COMPACT_FOOTPRINT_AREA_M2
        else FULL_THERMAL_RADIUS_FACTOR
    )
    return factor * params.collapse.blast_radius_m


def _trail_surface_positions(positions: np.ndarray, swarm: Swarm) -> np.ndarray:
    """Convert chassis-centre positions to the route's surface coordinates."""
    surface = np.asarray(positions, dtype=np.float64).copy()
    for index, member in enumerate(swarm.members):
        surface[index, 2] -= member.assignment.spec.ride_height_m
    return surface


def _local_disaster_assignment(assignment: Assignment, footprint=None) -> Assignment:
    """Turn the route prefix into a visible mission several metres indoors."""

    if len(assignment.route) == 0:
        return assignment
    start = np.asarray(assignment.spawn, dtype=np.float64).copy()
    start[2] -= assignment.spec.ride_height_m
    stop = LIVE_DISASTER_ROUTE_WAYPOINTS
    if footprint is not None:
        depths = np.asarray(_interior_depths(assignment.route, footprint))
        indoors = np.flatnonzero(depths >= LIVE_DISASTER_MIN_INTERIOR_DEPTH_M)
        # Show several certified moves after the facade crossing instead of
        # cutting a compact route while it is still circling outside.
        stop = max(stop, int(indoors[0]) + 9) if indoors.size else len(assignment.route)
    route = assignment.route[:stop]
    if len(route) and np.linalg.norm(route[0] - start) < 0.05:
        local_route = route
    else:
        local_route = np.vstack((start, route))
    return replace(
        assignment,
        route=local_route,
        goal_storey=0,
        goal_label="S0 · ENTRY SEARCH",
    )


def _fan_out_disaster_assignments(
    assignments: list[Assignment],
    graph: TraversabilityGraph,
    footprint,
) -> list[Assignment]:
    """Append short certified branches so indoor robots remain distinguishable."""

    routed = [assignment for assignment in assignments if len(assignment.route)]
    if not routed:
        return assignments
    positions = graph.world_positions()
    depth = np.asarray(_interior_depths(positions, footprint))
    chosen: list[int] = []
    result: list[Assignment] = []
    for assignment in assignments:
        if len(assignment.route) == 0:
            result.append(assignment)
            continue
        # A truncated route is not guaranteed to share its endpoint with any
        # other robot, especially in the compact world where a complete route
        # can be shorter than the large world's prefix.  Build every branch
        # from that robot's actual endpoint instead of borrowing robot 0's.
        anchor = nearest_node(graph, tuple(assignment.route[-1]))
        distance, previous = dijkstra_from(graph, anchor)
        base = (
            (distance >= 0.55)
            & (depth >= LIVE_DISASTER_MIN_INTERIOR_DEPTH_M)
            & np.isclose(positions[:, 2], positions[anchor, 2])
        )
        candidates = np.flatnonzero(base & (distance <= 2.0))
        available = candidates[~np.isin(candidates, chosen)]
        if available.size == 0:
            # Smaller buildings have fewer certified cells inside a two-metre
            # disc.  Widen locally rather than weakening the required interior
            # depth or inventing an off-graph destination.
            candidates = np.flatnonzero(base & (distance <= 4.0))
            available = candidates[~np.isin(candidates, chosen)]
        if available.size == 0:
            raise RuntimeError("not enough certified indoor cells to fan out the swarm")
        if chosen:
            separation = np.min(
                np.linalg.norm(
                    positions[available, :2][:, None, :]
                    - positions[chosen, :2][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            goal = int(available[int(np.argmax(separation))])
        else:
            goal = int(available[int(np.argmax(depth[available]))])
        chosen.append(goal)
        branch = path_from(distance, previous, anchor, goal)
        goal_position = positions[goal]
        result.append(
            replace(
                assignment,
                route=np.vstack((assignment.route, positions[branch[1:]])),
                goal_label=(f"S0 · X{goal_position[0]:.1f} Y{goal_position[1]:.1f}"),
            )
        )
    return result


def _interior_depths(positions: np.ndarray, footprint) -> tuple[float, ...]:
    """Signed distance from each robot to the nearest building facade."""

    return tuple(
        float(
            min(
                x - footprint.x0,
                footprint.x1 - x,
                y - footprint.y0,
                footprint.y1 - y,
            )
        )
        for x, y in np.asarray(positions)[:, :2]
    )


def _search_target_node(
    graph: TraversabilityGraph,
    estimate_xy: np.ndarray,
    footprint,
    floor_z: float,
) -> int:
    """Project a noisy BLE estimate onto a certified deep interior cell."""

    positions = graph.world_positions()
    depth = np.asarray(_interior_depths(positions, footprint))
    candidates = np.flatnonzero(
        (depth >= LIVE_DISASTER_MIN_INTERIOR_DEPTH_M)
        & np.isclose(positions[:, 2], floor_z)
    )
    if candidates.size == 0:
        raise RuntimeError("no certified ground-floor search cells remain")
    offset = positions[candidates, :2] - np.asarray(estimate_xy, dtype=np.float64)
    return int(candidates[int(np.argmin(np.linalg.norm(offset, axis=1)))])


def _baseline_route(
    graph: TraversabilityGraph, start_m: np.ndarray, goal_node: int
) -> np.ndarray:
    """Shortest physical route used as the pre-belief comparison."""

    start_node = nearest_node(graph, tuple(start_m))
    distance, previous = dijkstra_from(graph, start_node)
    nodes = path_from(distance, previous, start_node, goal_node)
    if not nodes:
        raise RuntimeError("the localized survivor site is unreachable")
    route = graph.world_positions()[nodes]
    if np.linalg.norm(route[0] - start_m) > 0.05:
        route = np.vstack((start_m, route))
    return route


def _routing_hazard_center(
    assignments: list[Assignment],
) -> tuple[float, float, float]:
    """Put a small sensed fire on the shared entry route, before its branches."""

    routed = [assignment for assignment in assignments if len(assignment.route)]
    if not routed:
        raise RuntimeError("a routing hazard needs at least one routed robot")
    route = routed[0].route
    if len(route) <= LIVE_ROUTING_HAZARD_WAYPOINT + 1:
        raise RuntimeError("the live route is too short for a routing dilemma")
    point = route[LIVE_ROUTING_HAZARD_WAYPOINT]
    return (float(point[0]), float(point[1]), LIVE_ROUTING_HAZARD_HEIGHT_M)


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
    live_disaster: bool = False
    failed_members: tuple[str, ...] = ()
    robot_progress: tuple[float, ...] = ()
    robot_displacement_m: tuple[float, ...] = ()
    robot_interior_depth_m: tuple[float, ...] = ()
    belief_reroutes: int = 0
    collapse_replans: int = 0
    collapse_retained: int = 0
    routing_observed_nodes: int = 0
    routing_sweeps: int = 0
    routing_residual: float = 0.0
    routing_converged: bool = False
    routing_cost_before: float = 0.0
    routing_cost_after: float = 0.0
    robot_route_indices: tuple[int, ...] = ()
    robot_positions_m: tuple[tuple[float, float, float], ...] = ()
    survivors_hidden: int = 0
    survivors_localized: int = 0
    max_localization_error_m: float = 0.0
    autonomous_search: bool = False
    distributed_physical: bool = False
    radio_bytes: int = 0
    radio_deliveries: int = 0
    radio_below_sensitivity: int = 0
    max_radio_wall_count: int = 0
    geometry_revisions: int = 0
    preserved_factors: int = 0
    peer_factors: int = 0
    numeric_arena_high_water_bytes: int = 0
    analytical_deadline_misses: int = 0
    repaired_spawns: int = 0
    planning_height_limit_m: float | None = None
    physical_arrival_radius_m: float = 0.4
    physical_deadline_interventions: int = 0
    modeled_deadline_infeasible_plans: int = 0

    @property
    def arrived(self) -> int:
        return sum(1 for _, _, _, status in self.per_robot if status == "arrived")

    @property
    def succeeded(self) -> bool:
        """Every deployed robot arrived, with at least one reaching upstairs."""
        upper = {s for s in self.storeys_reached if s >= 1}
        return (
            bool(upper)
            and self.n_robots > 0
            and len(self.per_robot) == self.n_robots
            and all(status == "arrived" for _, _, _, status in self.per_robot)
        )

    @property
    def visibly_moved(self) -> int:
        return sum(
            distance >= LIVE_DISASTER_MIN_TRAVEL_M
            for distance in self.robot_displacement_m
        )

    @property
    def inside(self) -> int:
        return sum(
            depth >= LIVE_DISASTER_MIN_INTERIOR_DEPTH_M
            for depth in self.robot_interior_depth_m
        )

    @property
    def accepted(self) -> bool:
        if not self.live_disaster:
            return self.succeeded
        if self.distributed_physical:
            return (
                bool(self.failed_members)
                and self.n_robots >= 3
                and self.arrived == self.n_robots
                and self.inside == self.n_robots
                and self.routing_converged
                and self.autonomous_search
                and self.survivors_localized == self.survivors_hidden == self.n_robots
                and self.max_localization_error_m <= LIVE_SEARCH_MAX_ERROR_M
                and self.geometry_revisions >= 1
                and self.peer_factors > 0
            )
        return (
            bool(self.failed_members)
            and self.n_robots >= 2
            and self.visibly_moved >= 2
            and self.arrived == self.n_robots
            and self.inside == self.n_robots
            and self.belief_reroutes == self.n_robots
            and self.collapse_replans >= 2
            and self.collapse_replans + self.collapse_retained == self.n_robots
            and self.routing_converged
            and self.routing_cost_after < self.routing_cost_before
            and self.autonomous_search
            and self.survivors_hidden == self.n_robots
            and self.survivors_localized == self.survivors_hidden
            and self.max_localization_error_m <= LIVE_SEARCH_MAX_ERROR_M
        )

    def report(self) -> str:
        lines = [
            f"seed {self.seed}: {self.arrived}/{self.n_robots} robots arrived",
            f"  acceptance             : {'PASS' if self.accepted else 'FAIL'}",
            f"  planner says reachable : {sorted(self.planner_reachable)}",
            f"  swarm actually visited : {sorted(self.storeys_reached)}",
            f"  simulated              : {self.steps} steps "
            f"({self.steps * 0.002:.1f} s)",
        ]
        for name, storey, goal, status in self.per_robot:
            lines.append(f"    {name:<8} storey {storey} -> {goal}   {status}")
        if self.robot_progress:
            lines.append(
                "  route completion       : "
                + ", ".join(f"{progress:.0%}" for progress in self.robot_progress)
            )
        for index, (_, _, _, status) in enumerate(self.per_robot):
            if status == "arrived" or index >= len(self.robot_positions_m):
                continue
            position = self.robot_positions_m[index]
            route_index = (
                self.robot_route_indices[index]
                if index < len(self.robot_route_indices)
                else -1
            )
            lines.append(
                f"  failure detail         : robot{index} waypoint {route_index} "
                f"at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
            )
        if self.live_disaster:
            lines.append(
                f"  autonomous BLE search  : "
                f"{self.survivors_localized}/{self.survivors_hidden} localized · "
                f"max error {self.max_localization_error_m:.2f} m"
            )
            lines.append(
                f"  thermal failure        : {len(self.failed_members)} members"
            )
            lines.append(
                f"  robots visibly moved   : {self.visibly_moved}/{self.n_robots}"
            )
            lines.append(
                "  final displacement (m) : "
                + ", ".join(f"{distance:.2f}" for distance in self.robot_displacement_m)
            )
            lines.append(f"  agents deep inside     : {self.inside}/{self.n_robots}")
            lines.append(
                f"  belief-driven reroutes : {self.belief_reroutes} "
                f"({self.routing_observed_nodes} local cells observed)"
            )
            lines.append(
                f"  post-collapse replans  : {self.collapse_replans}/{self.n_robots}"
            )
            if self.collapse_retained:
                lines.append(
                    f"  safe routes retained   : "
                    f"{self.collapse_retained}/{self.n_robots}"
                )
            lines.append(
                f"  routing solves         : max {self.routing_sweeps} sweeps · "
                f"max residual {self.routing_residual:.2e} · "
                f"{'converged' if self.routing_converged else 'not converged'}"
            )
            lines.append(
                "  posterior route cost   : "
                f"{self.routing_cost_before:.3f} -> "
                f"{self.routing_cost_after:.3f}"
            )
        if self.video:
            lines.append(f"  video                  : {self.video}")
        if self.distributed_physical:
            lines.extend(
                (
                    f"  distributed radio      : {self.radio_bytes} bytes · "
                    f"{self.radio_deliveries} deliveries",
                    f"  geometry rebuilds      : {self.geometry_revisions} · "
                    f"{self.preserved_factors} factors preserved",
                    f"  delivered peer factors : {self.peer_factors}",
                    f"  numeric arena peak     : {self.numeric_arena_high_water_bytes} "
                    "bytes (host buffers only)",
                    f"  analytical misses      : {self.analytical_deadline_misses} "
                    "(not calibrated timing)",
                )
            )
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
    return collapse, settled_poses(dynamic, settled, structure)


def _thermal_failure_zone(
    structure: Structure, collapse: CollapseResult, radius_m: float
):
    """Place demo ignition around a column that survived the first collapse."""

    destroyed = set(collapse.damage.destroyed_columns)
    candidates = [
        member
        for member in structure.members
        if member.kind is MemberKind.COLUMN
        and member.storey == 0
        and member.name not in destroyed
    ]
    if not candidates:
        raise RuntimeError("the initial collapse left no column for a live disaster")
    cx, cy, _ = collapse.damage.center
    target = max(
        candidates,
        key=lambda member: (member.center[0] - cx) ** 2 + (member.center[1] - cy) ** 2,
    )
    return compute_damage(structure, target.center, radius_m)


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
    live_disaster: bool = False,
    distributed_physical: bool = False,
) -> DemoResult:
    if live_disaster and n_robots < 3:
        raise ValueError("live autonomous search requires at least three robots")
    if distributed_physical and not live_disaster:
        raise ValueError("distributed physical mode requires a live-disaster mission")
    if distributed_physical:
        params = collision_aware_params(params, RobotSpec())
    building = params.building
    plan = build_floorplan(seed, building)
    structure = build_structure(plan)
    collapse, poses = _settled_scene(seed, params, plan, structure, use_cache)

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
        margin_m=4.0 if distributed_physical else GRID_MARGIN_M,
    )
    graph = build_graph(grid, params.traversability, bare, bare_data)
    if distributed_physical:
        graph = ground_mission_graph(graph)
    print(
        f"[cascade] traversability graph: {graph.n_nodes} cells, "
        f"{graph.n_edges} moves, occupancy {grid.occupancy_fraction():.3f}"
    )

    assignments = plan_swarm(plan, graph, building, n_robots, RobotSpec())
    if live_disaster:
        assignments = _fan_out_disaster_assignments(
            [
                _local_disaster_assignment(assignment, plan.footprint)
                for assignment in assignments
            ],
            graph,
            plan.footprint,
        )
    routing_hazard_center = (
        _routing_hazard_center(assignments) if live_disaster else None
    )
    survivor_sites = np.empty((0, 3), dtype=np.float64)
    if live_disaster:
        entry = nearest_node(
            graph,
            (plan.exits[0].center[0], plan.exits[0].center[1], 0.05),
        )
        survivor_sites = sample_survivor_sites(
            graph,
            entry,
            n_robots,
            (
                plan.footprint.x0,
                plan.footprint.y0,
                plan.footprint.x1,
                plan.footprint.y1,
            ),
            stage_rng(seed, "survivors"),
            minimum_depth_m=LIVE_DISASTER_MIN_INTERIOR_DEPTH_M,
            minimum_route_cost=LIVE_SEARCH_MIN_ROUTE_COST,
            maximum_route_cost=LIVE_SEARCH_MAX_ROUTE_COST,
        )
        print(
            f"[ember] {len(survivor_sites)} hidden survivor beacons placed "
            "procedurally; coordinates withheld from agents"
        )
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
    final_xml = (
        to_mjcf(
            structure,
            timestep_s=params.collapse.timestep_s,
            extra_worldbody=bodies,
            extra_actuators=actuators,
        )
        if live_disaster
        else to_settled_mjcf(
            structure,
            poses,
            extra_worldbody=bodies,
            extra_actuators=actuators,
        )
    )
    model = mujoco.MjModel.from_xml_string(final_xml)
    data = mujoco.MjData(model)
    if live_disaster:
        apply_settled_state(
            model,
            data,
            collapse,
            allow_appended_joints=True,
        )
    else:
        mujoco.mj_forward(model, data)

    swarm = (
        Swarm(
            assignments,
            departure_stagger_steps=LIVE_DISASTER_STAGGER_STEPS,
            cruise_rad_s=LIVE_DISASTER_CRUISE_RAD_S,
        )
        if live_disaster
        else Swarm(assignments)
    )
    swarm.place(model, data)
    repaired_spawns = 0
    if distributed_physical:
        for member in swarm.members:
            member.driver.arrive_radius_m = min(
                member.driver.arrive_radius_m,
                0.4 * params.traversability.voxel_size_m,
            )
        repaired_spawns = place_collision_free_team(
            swarm, model, data, graph, plan.footprint
        )
        routing_hazard_center = _routing_hazard_center(
            [m.assignment for m in swarm.members]
        )
        print(
            f"[ember] collision-free deployment: {repaired_spawns} "
            "penetrating spawns repaired"
        )
    print(
        f"[cascade] settled scene: {len(poses)} members standing, "
        f"{len(swarm.members)} robots, {model.nbody - 1} dynamic bodies"
    )

    # Do not release a robot on a fixed clock.  It may still be bouncing at the
    # old 0.5 s deadline, especially after landing on rubble near an entrance.
    settle_steps, settled = swarm.settle(model, data)
    _check_stable(model, data, strict=distributed_physical)
    if not settled:
        raise RuntimeError(f"swarm did not settle within {settle_steps} physics steps")
    print(f"[cascade] swarm settled in {settle_steps} steps")
    start_positions = swarm.positions(data).copy()
    physical_team = (
        PhysicalRobotTeam(
            graph,
            _trail_surface_positions(start_positions, swarm),
            loss_seed=int(stage_rng(seed, "radio_loss").integers(0, 2**63)),
        )
        if distributed_physical
        else None
    )

    thermal_zone = (
        _thermal_failure_zone(
            structure,
            collapse,
            _thermal_damage_radius(params),
        )
        if live_disaster
        else None
    )
    thermal_domain = HazardDomain.from_floorplan(plan) if live_disaster else None
    thermal_state = None
    if thermal_domain is not None and thermal_zone is not None:
        thermal_state = thermal_domain.initial_state().ignite(
            thermal_domain,
            thermal_zone.center,
            radius_m=thermal_zone.radius_m,
            temperature_k=1100.0,
            smoke_kg_m3=0.25,
        )
        assert routing_hazard_center is not None
        thermal_state = thermal_state.ignite(
            thermal_domain,
            routing_hazard_center,
            radius_m=LIVE_ROUTING_HAZARD_RADIUS_M,
            temperature_k=1100.0,
            smoke_kg_m3=0.25,
        )
    thermal_solver = (
        HazardSolver(thermal_domain) if thermal_domain is not None else None
    )
    thermal_damage = (
        ThermalDamage(
            structure,
            ThermalDamageConfig(failure_exposure_k_s=LIVE_THERMAL_EXPOSURE_K_S),
        )
        if live_disaster
        else None
    )
    stage = Stage(
        seed=seed,
        building=building,
        groups=index_geoms(model, data),
        camera=SwarmCamera(footprint_m=building.footprint_m),
        trails=Trails(n_robots=len(swarm.members)),
        aspect=width / height,
        disaster_focus=thermal_zone.center if thermal_zone is not None else None,
        disaster_at_s=LIVE_DISASTER_AT_S,
        routing_hazard_focus=routing_hazard_center,
        graph_positions=graph.world_positions(),
        distributed_physical=distributed_physical,
    )
    every = max(1, round(speed / (fps * params.collapse.timestep_s)))

    renderer = (
        mujoco.Renderer(model, height=height, width=width, max_geom=MAX_SCENE_GEOMS)
        if render
        else None
    )
    writer = None
    step = 0
    video = None
    frames = 0
    thermal_failed: set[str] = set()
    routing_results: list[PhysicalRouteResult] = []
    collapse_routing_results: list[PhysicalRouteResult] = []
    routing_rng = stage_rng(seed, "hazard")
    search_rng = stage_rng(seed, "search")
    search_result: CooperativeSearchResult | None = None
    search_targets: list[np.ndarray | None] = [None] * len(swarm.members)
    pending_geometry_refresh_s: float | None = None
    hazard_every = max(1, round(LIVE_HAZARD_DT_S / params.collapse.timestep_s))
    drive_limit = (
        round(LIVE_DISASTER_SECONDS / params.collapse.timestep_s)
        if live_disaster
        else MAX_DRIVE_STEPS
    )

    try:
        writer = (
            VideoWriter(
                out_dir
                / (
                    f"seed{seed}_end_to_end.mp4"
                    if live_disaster
                    else f"seed{seed}_swarm.mp4"
                ),
                fps=fps,
            )
            if render
            else None
        )
        while step < drive_limit and (live_disaster or not swarm.done):
            if live_disaster and step % hazard_every == 0:
                assert thermal_state is not None
                assert thermal_solver is not None
                assert thermal_domain is not None
                assert thermal_damage is not None
                thermal_state = thermal_solver.step(thermal_state)
                elapsed_s = step * params.collapse.timestep_s
                if not routing_results and elapsed_s >= LIVE_SEARCH_LOCALIZE_AT_S:
                    robot_positions = _trail_surface_positions(
                        swarm.positions(data), swarm
                    )
                    search_result = localize_and_assign(
                        robot_positions[:, :2],
                        survivor_sites[:, :2],
                        search_rng,
                        config=RSSIConfig(noise_std_db=0.15),
                    )
                    stage.search_result = search_result
                    stage.localization_robot_positions = robot_positions.copy()
                    stage.search_status = (
                        f"BLE SEARCH · {len(survivor_sites)}/{len(survivor_sites)} "
                        f"LOCALIZED · MAX ERROR {search_result.max_error_m:.2f} m"
                    )
                    if physical_team is not None:
                        for index in range(len(swarm.members)):
                            beacon = search_result.robot_to_beacon[index]
                            target_node = _search_target_node(
                                graph,
                                search_result.estimated_positions_m[beacon],
                                plan.footprint,
                                survivor_sites[beacon, 2],
                            )
                            search_targets[index] = graph.world_positions()[target_node]
                        physical_team.observe_and_exchange(
                            thermal_domain,
                            thermal_state,
                            robot_positions,
                            np.asarray(search_targets),
                            routing_rng,
                            model=model,
                            data=data,
                            scan_radius_m=LIVE_ROUTING_SCAN_RADIUS_M,
                        )
                    for index, member in enumerate(swarm.members):
                        surface_position = np.asarray(
                            data.xpos[member.body_id], dtype=np.float64
                        ).copy()
                        surface_position[2] -= member.assignment.spec.ride_height_m
                        beacon = search_result.robot_to_beacon[index]
                        target_node = _search_target_node(
                            graph,
                            search_result.estimated_positions_m[beacon],
                            plan.footprint,
                            survivor_sites[beacon, 2],
                        )
                        target = graph.world_positions()[target_node]
                        search_targets[index] = target.copy()
                        baseline = _baseline_route(graph, surface_position, target_node)
                        result = (
                            physical_team.plan(
                                index,
                                thermal_domain,
                                thermal_state,
                                surface_position,
                                target,
                                baseline,
                                routing_rng,
                                available_time_s=max(
                                    0, LIVE_DISASTER_SECONDS - elapsed_s - 2
                                ),
                            )
                            if physical_team is not None
                            else plan_belief_route(
                                graph,
                                thermal_domain,
                                thermal_state,
                                surface_position,
                                target,
                                baseline,
                                routing_rng,
                                scan_radius_m=LIVE_ROUTING_SCAN_RADIUS_M,
                            )
                        )
                        routing_results.append(result)
                        stage.route_results[member.name] = result
                        stage.initial_route_results[member.name] = result
                        stage.rejected_routes[member.name] = baseline.copy()
                        if result.converged and (
                            result.improved or physical_team is not None
                        ):
                            swarm.reroute(
                                index,
                                result.waypoints,
                                goal_label=(
                                    f"BLE B{beacon + 1} · "
                                    f"X{target[0]:.1f} Y{target[1]:.1f}"
                                ),
                            )
                            stage.belief_reroutes += 1
                        print(
                            f"[ember] {member.name} belief route: "
                            f"{result.prior_route_cost:.3f} -> "
                            f"{result.posterior_route_cost:.3f} cost, "
                            f"{result.sweeps} sweeps"
                        )
                failed_now = (
                    thermal_damage.step(
                        thermal_state,
                        thermal_domain,
                        model,
                        data,
                        LIVE_HAZARD_DT_S,
                    )
                    if _thermal_damage_active(step, params.collapse.timestep_s)
                    else ()
                )
                new_failed = (
                    set(failed_now)
                    - set(collapse.damage.destroyed_columns)
                    - thermal_failed
                )
                thermal_failed.update(new_failed)
                stage.failed_member_names.update(new_failed)
                stage.released_weld_names.update(thermal_damage.released_welds)
                stage.damage_exposure_k_s = thermal_damage.peak_exposure_k_s
                stage.damage_threshold_k_s = thermal_damage.failure_threshold_k_s
                stage.hazard_peak_temperature_k = float(
                    np.max(thermal_state.temperature_k)
                )
                stage.hazard_peak_smoke_kg_m3 = float(np.max(thermal_state.smoke_kg_m3))
                if thermal_failed:
                    stage.disaster_status = (
                        f"LIVE THERMAL FAILURE · {len(thermal_failed)} MEMBERS FAILED"
                    )
                if (
                    physical_team is not None
                    and new_failed
                    and pending_geometry_refresh_s is None
                ):
                    pending_geometry_refresh_s = elapsed_s + 1.5
                if (
                    (
                        new_failed
                        if physical_team is None
                        else (
                            pending_geometry_refresh_s is not None
                            and elapsed_s >= pending_geometry_refresh_s
                        )
                    )
                    and search_result is not None
                    and not collapse_routing_results
                ):
                    stage.collapse_event_s = elapsed_s
                    stage.camera.reset()
                    replan_started = perf_counter()
                    if physical_team is not None:
                        graph = ground_mission_graph(
                            rebuild_physical_graph(
                                model,
                                data,
                                structure,
                                params,
                                margin_m=4.0,
                            )
                        )
                        physical_team.replace_graph(graph)
                        stage.geometry_revisions = physical_team.geometry_revisions
                        stage.graph_positions = graph.world_positions()
                        physical_team.observe_and_exchange(
                            thermal_domain,
                            thermal_state,
                            _trail_surface_positions(swarm.positions(data), swarm),
                            np.asarray(search_targets),
                            routing_rng,
                            model=model,
                            data=data,
                            scan_radius_m=LIVE_COLLAPSE_SCAN_RADIUS_M,
                            beta=2.0,
                        )
                        pending_geometry_refresh_s = None
                        print(
                            f"[ember] current geometry rebuilt: {graph.n_nodes} cells, "
                            f"{graph.n_edges} moves"
                        )
                    for index, member in enumerate(swarm.members):
                        target = search_targets[index]
                        if target is None:
                            continue
                        route_index = min(
                            member.driver.index,
                            len(member.assignment.route) - 1,
                        )
                        previous = member.assignment.route[route_index:]
                        if member.driver.finished or (
                            len(previous) <= 8 and physical_team is None
                        ):
                            stage.collapse_retained += 1
                            stage.retained_robot_names.add(member.name)
                            print(
                                f"[ember] {member.name} collapse response: "
                                "remaining route already clear; retained"
                            )
                            continue
                        surface_position = np.asarray(
                            data.xpos[member.body_id], dtype=np.float64
                        ).copy()
                        surface_position[2] -= member.assignment.spec.ride_height_m
                        if np.linalg.norm(previous[0] - surface_position) > 0.05:
                            previous = np.vstack((surface_position, previous))
                        result = (
                            physical_team.plan(
                                index,
                                thermal_domain,
                                thermal_state,
                                surface_position,
                                target,
                                previous,
                                routing_rng,
                                beta=2.0,
                                available_time_s=max(
                                    0, LIVE_DISASTER_SECONDS - elapsed_s - 2
                                ),
                            )
                            if physical_team is not None
                            else plan_belief_route(
                                graph,
                                thermal_domain,
                                thermal_state,
                                surface_position,
                                target,
                                previous,
                                routing_rng,
                                scan_radius_m=LIVE_COLLAPSE_SCAN_RADIUS_M,
                                beta=2.0,
                            )
                        )
                        collapse_routing_results.append(result)
                        stage.route_results[member.name] = result
                        stage.rejected_routes[member.name] = previous.copy()
                        if result.converged:
                            swarm.reroute(
                                index,
                                result.waypoints,
                                goal_label=(
                                    f"B{search_result.robot_to_beacon[index] + 1} "
                                    "· SAFE ROUTE"
                                ),
                            )
                            stage.collapse_replans += 1
                        print(
                            f"[ember] {member.name} collapse response: "
                            f"route replaced in {result.sweeps} sweeps"
                        )
                    stage.replan_latency_ms = 1000.0 * (perf_counter() - replan_started)
            swarm.step(model, data)
            mujoco.mj_step(model, data)
            if distributed_physical:
                _check_stable(model, data, strict=True)
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
        if renderer is not None and writer is not None:
            terminal = stage.compose(
                model,
                data,
                renderer,
                swarm,
                elapsed_s=step * params.collapse.timestep_s,
                mission_status=(
                    _disaster_mission_status(
                        swarm,
                        len(thermal_failed),
                        sum(
                            depth >= LIVE_DISASTER_MIN_INTERIOR_DEPTH_M
                            for depth in _interior_depths(
                                swarm.positions(data), plan.footprint
                            )
                        ),
                        stage.belief_reroutes,
                        stage.collapse_replans,
                        stage.collapse_retained,
                    )
                    if live_disaster
                    else _mission_status(
                        swarm,
                        timed_out=step >= drive_limit and not swarm.done,
                    )
                ),
            )
            _append_hold(writer, terminal, fps=fps, seconds=FINAL_HOLD_S)
    finally:
        if renderer is not None:
            renderer.close()
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
        live_disaster=live_disaster,
        failed_members=tuple(sorted(thermal_failed)),
        robot_progress=tuple(member.progress for member in swarm.members),
        robot_displacement_m=tuple(
            float(distance)
            for distance in np.linalg.norm(
                swarm.positions(data)[:, :2] - start_positions[:, :2], axis=1
            )
        ),
        robot_interior_depth_m=_interior_depths(swarm.positions(data), plan.footprint),
        belief_reroutes=stage.belief_reroutes,
        collapse_replans=stage.collapse_replans,
        collapse_retained=stage.collapse_retained,
        routing_observed_nodes=sum(result.observed_nodes for result in routing_results),
        routing_sweeps=max((result.sweeps for result in routing_results), default=0),
        routing_residual=max(
            (result.residual for result in routing_results), default=0.0
        ),
        routing_converged=(
            len(routing_results) == len(swarm.members)
            and all(result.converged for result in routing_results)
        ),
        routing_cost_before=sum(result.prior_route_cost for result in routing_results),
        routing_cost_after=sum(
            result.posterior_route_cost for result in routing_results
        ),
        robot_route_indices=tuple(member.driver.index for member in swarm.members),
        robot_positions_m=tuple(
            tuple(float(value) for value in position)
            for position in swarm.positions(data)
        ),
        survivors_hidden=len(survivor_sites),
        survivors_localized=(
            len(search_result.estimated_positions_m) if search_result is not None else 0
        ),
        max_localization_error_m=(
            search_result.max_error_m if search_result is not None else 0.0
        ),
        autonomous_search=search_result is not None,
        distributed_physical=distributed_physical,
        radio_bytes=physical_team.radio_stats.bytes_air if physical_team else 0,
        repaired_spawns=repaired_spawns,
        planning_height_limit_m=1.0 if distributed_physical else None,
        physical_arrival_radius_m=max(m.driver.arrive_radius_m for m in swarm.members),
        physical_deadline_interventions=sum(
            r.deadline_intervened for r in (*routing_results, *collapse_routing_results)
        ),
        modeled_deadline_infeasible_plans=sum(
            r.modeled_deadline_feasible is False
            for r in (*routing_results, *collapse_routing_results)
        ),
        radio_deliveries=physical_team.radio_stats.delivered if physical_team else 0,
        radio_below_sensitivity=physical_team.radio_stats.below_sensitivity
        if physical_team
        else 0,
        max_radio_wall_count=physical_team.max_wall_count if physical_team else 0,
        geometry_revisions=physical_team.geometry_revisions if physical_team else 0,
        preserved_factors=physical_team.preserved_factors if physical_team else 0,
        peer_factors=sum(
            sender != robot.robot_id
            for robot in physical_team.robots
            for sender, _, _ in robot.seen_factors
        )
        if physical_team
        else 0,
        numeric_arena_high_water_bytes=physical_team.arena_high_water_bytes
        if physical_team
        else 0,
        analytical_deadline_misses=sum(r.deadline_misses for r in physical_team.robots)
        if physical_team
        else 0,
    )


def _check_stable(model: mujoco.MjModel, data: mujoco.MjData, *, strict=False) -> None:
    """Report a diverged solver rather than driving a broken simulation."""
    bad = int(data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
    if bad:
        if strict:
            raise RuntimeError(
                f"physical validation rejected {bad} bad-acceleration events"
            )
        print(
            f"[cascade] WARNING: solver reported {bad} bad-acceleration events "
            "during settling; a robot may have spawned inside geometry",
            file=sys.stderr,
        )
    if not np.all(np.isfinite(data.qpos)):
        raise RuntimeError("simulation diverged during settling")


def _mission_status(swarm: Swarm, *, timed_out: bool) -> str:
    """Short, explicit verdict for the held terminal frame."""
    arrived = sum(member.status == "arrived" for member in swarm.members)
    total = len(swarm.members)
    if total > 0 and arrived == total:
        return f"MISSION COMPLETE · {arrived}/{total} ARRIVED"
    if timed_out:
        return f"RUN FAILED · TIME LIMIT · {arrived}/{total} ARRIVED"
    failed = next(
        (member for member in swarm.members if member.status != "arrived"),
        None,
    )
    detail = (
        f"{failed.name.upper()} {failed.status.upper()}"
        if failed is not None
        else "NO AGENTS"
    )
    return f"RUN FAILED · {detail} · {arrived}/{total} ARRIVED"


def _disaster_mission_status(
    swarm: Swarm,
    failed_members: int,
    inside: int,
    belief_reroutes: int,
    collapse_replans: int,
    collapse_retained: int,
) -> str:
    """Verdict that keeps both the rescue and destruction legible."""

    arrived = sum(member.status == "arrived" for member in swarm.members)
    verdict = (
        "ENTRY MISSION COMPLETE"
        if arrived == len(swarm.members)
        and inside == len(swarm.members)
        and failed_members > 0
        and belief_reroutes == len(swarm.members)
        and collapse_replans >= 2
        and collapse_replans + collapse_retained == len(swarm.members)
        else "ENTRY MISSION FAILED"
    )
    return (
        f"{verdict} · {inside}/{len(swarm.members)} INSIDE · "
        f"{failed_members} STRUCTURAL MEMBERS FAILED · {collapse_replans} REPLANNED · "
        f"{collapse_retained} ROUTES RETAINED"
    )


def _draw_disaster_marker(
    scene,
    focus: tuple[float, float, float],
    since_impact_s: float,
) -> None:
    """Mark the simulator's hot structural zone."""

    if scene.ngeom >= scene.maxgeom or since_impact_s > 1.2:
        return
    radius = (
        0.65 if since_impact_s < 0.0 else 0.65 + 2.4 * min(1.0, since_impact_s / 0.35)
    )
    rgba = (
        (1.0, 0.55, 0.08, 0.8)
        if since_impact_s < 0.0
        else (1.0, 0.12, 0.02, max(0.15, 0.85 - 0.55 * since_impact_s))
    )
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, radius),
        np.asarray(focus, dtype=np.float64),
        np.eye(3).ravel(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _draw_routing_hazard_marker(
    scene,
    focus: tuple[float, float, float],
) -> None:
    """Keep the locally sensed fire visible in the tactical pane."""

    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, LIVE_ROUTING_HAZARD_RADIUS_M),
        np.asarray(focus, dtype=np.float64),
        np.eye(3).ravel(),
        np.asarray((1.0, 0.08, 0.02, 0.22), dtype=np.float32),
    )
    scene.ngeom += 1


def _draw_segment(scene, start, end, rgba, width: float = 0.025) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).ravel(),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        width,
        np.ascontiguousarray(start, dtype=np.float64),
        np.ascontiguousarray(end, dtype=np.float64),
    )
    scene.ngeom += 1


def _draw_ring(
    scene,
    centre_xy: np.ndarray,
    transform: np.ndarray,
    z: float,
    rgba,
    *,
    segments: int = 32,
    width: float = 0.022,
) -> None:
    angle = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    unit = np.column_stack((np.cos(angle), np.sin(angle)))
    xy = centre_xy + unit @ transform.T
    points = np.column_stack((xy, np.full(len(xy), z)))
    for start, end in pairwise(points):
        _draw_segment(scene, start, end, rgba, width)


def _draw_localization_evidence(
    scene,
    result: CooperativeSearchResult,
    robot_positions: np.ndarray,
    subject_index: int,
    floor_z: float,
) -> None:
    """Show noisy BLE constraints, fused uncertainty, and held-out truth."""

    beacon = result.robot_to_beacon[subject_index]
    ranges = rssi_to_range(
        result.rssi_dbm[:, beacon],
        config=RSSIConfig(noise_std_db=0.15),
    )
    z = floor_z + 0.12
    for centre, radius in zip(robot_positions[:, :2], ranges, strict=True):
        _draw_ring(
            scene,
            centre,
            np.eye(2) * radius,
            z,
            (0.25, 0.70, 1.0, 0.16),
            segments=24,
            width=0.012,
        )

    covariance = result.covariance_m2[beacon]
    values, vectors = np.linalg.eigh(covariance)
    ellipse = vectors @ np.diag(np.sqrt(5.991 * np.maximum(values, 0.0)))
    estimate = result.estimated_positions_m[beacon]
    _draw_ring(scene, estimate, ellipse, z + 0.025, (0.10, 0.95, 1.0, 0.95), width=0.04)

    truth = result.truth_positions_m[beacon]
    arm = 0.22
    _draw_segment(
        scene,
        (truth[0] - arm, truth[1] - arm, z + 0.04),
        (truth[0] + arm, truth[1] + arm, z + 0.04),
        (1.0, 1.0, 1.0, 0.95),
        0.035,
    )
    _draw_segment(
        scene,
        (truth[0] - arm, truth[1] + arm, z + 0.04),
        (truth[0] + arm, truth[1] - arm, z + 0.04),
        (1.0, 1.0, 1.0, 0.95),
        0.035,
    )


def _draw_belief_evidence(
    scene,
    graph_positions: np.ndarray,
    result: PhysicalRouteResult,
    z_range: tuple[float, float],
) -> None:
    """Render the tracked robot local mu + beta-sigma field, not oracle truth."""

    observed = result.observed_node_indices
    if result.graph_positions is not None:
        graph_positions = result.graph_positions
    if observed.size == 0:
        return
    stride = max(1, observed.size // 220)
    nodes = observed[::stride]
    points = graph_positions[nodes]
    visible = (points[:, 2] >= z_range[0]) & (points[:, 2] < z_range[1])
    nodes = nodes[visible]
    points = points[visible]
    if not len(nodes):
        return
    values = result.risk_cost[nodes]
    low = float(np.min(values))
    high = float(np.quantile(values, 0.95))
    scale = np.clip((values - low) / max(1.0e-9, high - low), 0.0, 1.0)
    for point, level in zip(points, scale, strict=True):
        if scene.ngeom >= scene.maxgeom:
            return
        geom = scene.geoms[scene.ngeom]
        colour = (0.10 + 0.90 * level, 0.55 * (1.0 - level), 1.0 - level, 0.32)
        centre = np.asarray(point, dtype=np.float64).copy()
        centre[2] += 0.045
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.asarray((0.16, 0.16, 0.025)),
            centre,
            np.eye(3).ravel(),
            np.asarray(colour, dtype=np.float32),
        )
        scene.ngeom += 1


def _draw_robot_halos(
    scene,
    positions: np.ndarray,
    liveries: list[tuple[float, float, float]],
    subject_index: int,
    floor_z: float,
) -> None:
    for index, (position, colour) in enumerate(zip(positions, liveries, strict=True)):
        radius = 0.42 if index == subject_index else 0.29
        _draw_ring(
            scene,
            position[:2],
            np.eye(2) * radius,
            floor_z + 0.16,
            (*colour, 1.0),
            segments=18,
            width=0.045 if index == subject_index else 0.025,
        )


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
    disaster_status: str | None = None
    disaster_focus: tuple[float, float, float] | None = None
    disaster_at_s: float = 0.0
    routing_hazard_focus: tuple[float, float, float] | None = None
    search_status: str | None = None
    belief_reroutes: int = 0
    collapse_replans: int = 0
    collapse_retained: int = 0
    collapse_event_s: float | None = None
    retained_robot_names: set[str] = field(default_factory=set)
    hazard_peak_temperature_k: float | None = None
    hazard_peak_smoke_kg_m3: float | None = None
    graph_positions: np.ndarray | None = None
    search_result: CooperativeSearchResult | None = None
    localization_robot_positions: np.ndarray | None = None
    route_results: dict[str, PhysicalRouteResult] = field(default_factory=dict)
    initial_route_results: dict[str, PhysicalRouteResult] = field(default_factory=dict)
    rejected_routes: dict[str, np.ndarray] = field(default_factory=dict)
    failed_member_names: set[str] = field(default_factory=set)
    released_weld_names: set[str] = field(default_factory=set)
    damage_exposure_k_s: float = 0.0
    damage_threshold_k_s: float = LIVE_THERMAL_EXPOSURE_K_S
    replan_latency_ms: float | None = None
    distributed_physical: bool = False
    geometry_revisions: int = 0
    _subject_name: str | None = field(default=None, init=False, repr=False)
    _shot_started_s: float = field(default=0.0, init=False, repr=False)
    _terminal_since_s: float | None = field(default=None, init=False, repr=False)

    def compose(
        self,
        model,
        data,
        renderer,
        swarm: Swarm,
        *,
        elapsed_s: float,
        mission_status: str | None = None,
    ) -> np.ndarray:
        """Render the evidence-first research view and its paced cards."""
        positions = swarm.positions(data)
        self.trails.record(_trail_surface_positions(positions, swarm))
        subject = self._choose_subject(swarm, elapsed_s)
        subject_at = np.asarray(data.xpos[subject.body_id], dtype=np.float64)
        storey, floor_surface_z, floor_z, plan_ceiling = self._slice(
            float(subject_at[2])
        )
        disaster = self.disaster_focus is not None
        collapse_shot = disaster and self._show_collapse(elapsed_s)
        ceiling = (
            self.building.total_height_m + CEILING_MARGIN_M
            if collapse_shot
            else floor_surface_z + CEILING_MARGIN_M
        )
        traces = self._traces(swarm)
        deployed = sum(
            member.state is not RobotState.WAITING for member in swarm.members
        )
        subject_index = next(
            index for index, member in enumerate(swarm.members) if member is subject
        )
        subject_trace = [traces[subject_index]]

        if collapse_shot:
            event_style(model, self.groups)
            highlight_members(model, self.failed_member_names)
        else:
            hero_style(model, self.groups, ceiling)
        hero_view = (
            self._disaster_view(subject_at)
            if collapse_shot
            else self.camera.update(
                subject_at,
                self._camera_companions(data, swarm, subject),
            )
        )
        hero = render_scene(
            model,
            data,
            hero_view,
            renderer=renderer,
            overlay=lambda scene: self._hero_overlay(
                scene,
                subject_trace,
                elapsed_s,
                None if disaster else (floor_z, plan_ceiling),
            ),
        )

        plan_style(model, data, self.groups, floor_z, plan_ceiling)
        top = render_scene(
            model,
            data,
            self._swarm_view(positions, floor_surface_z)
            if disaster
            else plan_view(
                self.building.footprint_m,
                floor_surface_z,
                aspect=self.aspect,
            ),
            renderer=renderer,
            overlay=lambda scene: self._plan_overlay(
                scene,
                subject_trace if disaster else traces,
                (floor_z, plan_ceiling),
                positions,
                [member.assignment.spec.livery for member in swarm.members],
                subject_index,
                floor_surface_z,
            ),
        )

        panes = stack_panes(
            hero,
            top,
            plan_fraction=0.65 if disaster else None,
            target_width=2 * hero.shape[1] if disaster else None,
            hero_label=(
                "FAILURE CAMERA · CAUSAL WELD RELEASE"
                if collapse_shot
                else (
                    f"EVIDENCE CAMERA · {subject.name.upper()} · "
                    f"{subject.status.upper()}"
                )
                if disaster
                else f"{subject.name.upper()} · {subject.status.upper()}"
            ),
            plan_label=(
                f"EVALUATION VIEW · FLOOR {storey} · N UP · RED DASH REJECTED · "
                "GREEN PLANNED · ROBOT COLOUR DRIVEN"
                if disaster
                else f"PLAN · STOREY {storey} · DESTINATIONS MARKED"
            ),
        )
        readouts = self._readouts(data, swarm, subject, elapsed_s)
        frame = draw_hud(
            panes,
            title="EMBER · PHASE 1 INTEGRATION" if disaster else "EMBER RESCUE DEMO",
            subtitle=(
                self._disaster_subtitle(elapsed_s)
                or f"Phase 0 — swarm traversal of a blast-collapsed "
                f"{self.building.n_storeys}-storey building · seed {self.seed}"
            ),
            readouts=readouts,
            elapsed_s=elapsed_s,
            n_storeys=self.building.n_storeys + 1,
            mission_status=(
                f"ILLUSTRATIVE SEED {self.seed}"
                if disaster and mission_status is None
                else mission_status
            ),
            show_storeys=not disaster,
            research_metrics=self._research_metrics(deployed, len(swarm.members))
            if disaster
            else None,
        )
        if disaster and mission_status is not None:
            return self._results_card(frame, readouts)
        if disaster and elapsed_s < 3.6:
            return research_card(
                frame,
                eyebrow=f"PHASE 1 INTEGRATION DEMO · ILLUSTRATIVE SEED {self.seed}",
                title="Can local observations reject a newly unsafe route?",
                body=(
                    "Independent robot GBP maps receive visibility-filtered scans "
                    "and delivered peer packets.\n"
                    "Integer LMDP routes drive the live robots."
                    if self.distributed_physical
                    else "Centralized Phase-1 reference integration: noisy BLE and "
                    "local hazard scans feed classical Jacobi/LMDP routing.\n"
                    "Distributed GBP/radio execution is not active in this clip."
                ),
                footer=(
                    "Complete planning graph is supplied to every route solve · "
                    "simulator hazard truth is evaluation-only"
                ),
            )
        return frame

    def _show_collapse(self, elapsed_s: float) -> bool:
        """Use the event camera only while the structural failure is legible."""

        if self.collapse_event_s is not None:
            return (
                self.collapse_event_s - 0.4 <= elapsed_s <= self.collapse_event_s + 4.0
            )
        return self.disaster_at_s - 0.4 <= elapsed_s <= self.disaster_at_s + 4.0

    def _disaster_view(self, subject_at: np.ndarray) -> CameraView:
        """Low event shot centered on the failing bay, not the empty roof."""

        assert self.disaster_focus is not None
        width, depth = self.building.footprint_m
        focus = np.asarray(self.disaster_focus, dtype=np.float64)
        look_xy = 0.82 * focus[:2] + 0.18 * subject_at[:2]
        return CameraView(
            name="live_disaster",
            lookat=(
                float(look_xy[0]),
                float(look_xy[1]),
                min(2.5, 0.28 * self.building.total_height_m),
            ),
            distance=1.15 * max(width, depth),
            azimuth=135.0,
            elevation=-30.0,
        )

    @staticmethod
    def _swarm_view(positions: np.ndarray, floor_z: float) -> CameraView:
        """Tight top-down shot containing every robot in the local mission."""

        centre = np.mean(positions[:, :2], axis=0)
        spread = float(np.max(np.linalg.norm(positions[:, :2] - centre, axis=1)))
        return CameraView(
            name="swarm_action",
            lookat=(float(centre[0]), float(centre[1]), floor_z),
            distance=min(18.0, max(10.5, 2.0 * spread + 7.0)),
            azimuth=90.0,
            elevation=-89.0,
        )

    def _hero_overlay(
        self,
        scene,
        traces: list[PathTrace],
        elapsed_s: float,
        z_range: tuple[float, float] | None,
    ) -> None:
        draw_paths(scene, traces, z_range=z_range)
        if self.disaster_focus is not None:
            _draw_disaster_marker(
                scene,
                self.disaster_focus,
                elapsed_s - (self.collapse_event_s or self.disaster_at_s),
            )

    def _plan_overlay(
        self,
        scene,
        traces: list[PathTrace],
        z_range: tuple[float, float],
        positions: np.ndarray,
        liveries: list[tuple[float, float, float]],
        subject_index: int,
        floor_z: float,
    ) -> None:
        result = self.route_results.get(self._subject_name or "")
        if result is not None and self.graph_positions is not None:
            _draw_belief_evidence(scene, self.graph_positions, result, z_range)
        if (
            self.search_result is not None
            and self.localization_robot_positions is not None
            and self.collapse_event_s is None
        ):
            _draw_localization_evidence(
                scene,
                self.search_result,
                self.localization_robot_positions,
                subject_index,
                floor_z,
            )
        draw_paths(scene, traces, z_range=z_range)
        _draw_robot_halos(scene, positions, liveries, subject_index, floor_z)
        if self.routing_hazard_focus is not None:
            _draw_routing_hazard_marker(scene, self.routing_hazard_focus)

    def _disaster_subtitle(self, elapsed_s: float) -> str | None:
        if self.disaster_focus is None:
            return self.disaster_status
        if self.search_status is None:
            return "STEP 1 OF 4 · DEPLOY AND SPREAD FOR BLE RANGING"
        if self.collapse_event_s is None:
            return "STEP 2 OF 4 · NOISY BLE -> FUSED ESTIMATE -> LOCAL BELIEF -> ROUTE"
        if elapsed_s <= self.collapse_event_s + 3.5:
            return (
                "STEP 3 OF 4 · TEMPERATURE -> DAMAGE INTEGRAL -> "
                "THRESHOLD -> WELD RELEASE"
            )
        return (
            f"STEP 4 OF 4 · {self.collapse_replans} NEW ROUTES · "
            f"{self.collapse_retained} ROUTE RETAINED · CONTINUE RESCUE"
        )

    def _research_metrics(self, deployed: int, total: int) -> list[ResearchMetric]:
        if self.search_result is None:
            return [
                ResearchMetric(
                    "Robots deployed", f"{deployed}/{total}", "local BLE ranging"
                ),
                ResearchMetric("Hidden targets", f"{total}", "coordinates withheld"),
                ResearchMetric(
                    "Planning graph", "FULL PRIOR", "supplied to every solve"
                ),
                ResearchMetric("Execution", "CENTRALIZED", "distributed path pending"),
            ]

        results = list(self.route_results.values())
        before = sum(result.baseline_proxy_cost for result in results)
        after = sum(result.planned_proxy_cost for result in results)
        sweeps = max((result.sweeps for result in results), default=0)
        max_radius = float(np.max(self.search_result.uncertainty_radii_m))
        if self.collapse_event_s is None:
            return [
                ResearchMetric(
                    "Localization error",
                    f"{self.search_result.max_error_m:.2f} m",
                    f"95% radius <= {max_radius:.2f} m",
                    (0, 210, 235),
                ),
                ResearchMetric(
                    "Agent objective", "mu - 2 sigma", "search policy; beta = -2"
                ),
                ResearchMetric(
                    "Observer proxy",
                    f"{before:.2f} -> {after:.2f}",
                    "same heat/smoke metric; pre-ISO",
                    (90, 220, 130),
                ),
                ResearchMetric(
                    "Jacobi solve", f"{sweeps} sweeps", "classical reference"
                ),
            ]

        ratio = self.damage_exposure_k_s / max(self.damage_threshold_k_s, 1.0e-9)
        decisions = self.collapse_replans + self.collapse_retained
        failed_member = min(self.failed_member_names, default="pending")
        released_weld = min(self.released_weld_names, default="pending")
        latency = (
            f"{self.replan_latency_ms:.1f} ms"
            if self.replan_latency_ms is not None
            else "pending"
        )
        return [
            ResearchMetric(
                "Thermal damage",
                f"{min(ratio, 9.99):.2f}x threshold",
                f"{failed_member} · {released_weld} released",
                (235, 101, 96),
            ),
            ResearchMetric(
                "Route decisions",
                f"{self.collapse_replans} / {self.collapse_retained}",
                f"replaced / retained · {decisions}/{total} checked",
                (90, 220, 130),
            ),
            ResearchMetric("Desktop reference", latency, f"max {sweeps} Jacobi sweeps"),
            ResearchMetric(
                "Observer proxy",
                f"{before:.2f} -> {after:.2f}",
                "same heat/smoke metric; pre-ISO",
            ),
        ]

    def _results_card(
        self, frame: np.ndarray, readouts: list[RobotReadout]
    ) -> np.ndarray:
        arrived = sum(readout.status == "arrived" for readout in readouts)
        total = len(readouts)
        results = list(self.initial_route_results.values())
        before = sum(result.baseline_proxy_cost for result in results)
        after = sum(result.planned_proxy_cost for result in results)
        change = 100.0 * (after - before) / before if before > 0.0 else 0.0
        error = (
            self.search_result.max_error_m if self.search_result is not None else 0.0
        )
        return research_card(
            frame,
            eyebrow=f"RESULTS · ILLUSTRATIVE SEED {self.seed}",
            title=f"{arrived}/{total} LOCALIZED TARGET SITES REACHED",
            body=(
                "Local and delivered observations changed routes; current structural "
                f"geometry was\nrebuilt {self.geometry_revisions} time(s) "
                "after thermal failure."
                if self.distributed_physical
                else "Local observations changed route objectives; a causal thermal "
                "failure triggered\nroute checks. Structural blockage is not "
                "rerasterized in this reference clip."
            ),
            metrics=[
                ResearchMetric(
                    "Localization", f"{error:.2f} m", "worst error", (0, 210, 235)
                ),
                ResearchMetric(
                    "Structure",
                    f"{len(self.failed_member_names)} members",
                    "thermally failed",
                    (235, 101, 96),
                ),
                ResearchMetric(
                    "Route response",
                    f"{self.collapse_replans} / {self.collapse_retained}",
                    "replaced / retained · all checked",
                    (90, 220, 130),
                ),
                ResearchMetric(
                    "Observer proxy delta",
                    f"{change:+.1f}%",
                    f"same pre-ISO metric · {before:.2f} -> {after:.2f}",
                ),
            ],
            footer=(
                "Distributed hazard maps · geometry-aware radio · integer routing · "
                "centralized BLE; ground-level search demo, "
                "no ISO survival or MCU timing validation."
                if self.distributed_physical
                else "Centralized reference · classical Jacobi · desktop timing · "
                "one illustrative seed; no FED, distributed, NCA, VoI, robot-failure, "
                "or MCU evidence."
            ),
        )

    def _storey(self, height_m: float) -> int:
        return self._slice(height_m)[0]

    def _slice(self, height_m: float) -> tuple[int, float, float, float]:
        """Storey and visibility band that always contain the subject."""
        storey_height = self.building.storey_height_m
        raw = math.floor((height_m + PLAN_FLOOR_DROP_M) / storey_height)
        storey = min(self.building.n_storeys, max(0, raw))
        surface = storey * storey_height
        floor = surface - PLAN_FLOOR_DROP_M
        ceiling = surface + storey_height - PLAN_FLOOR_DROP_M
        return storey, surface, floor, ceiling

    def _choose_subject(self, swarm: Swarm, elapsed_s: float):
        """Hold coherent shots, while immediately showing recovery events."""
        current = next(
            (m for m in swarm.members if m.name == self._subject_name),
            None,
        )
        active = [
            m for m in swarm.members if not (m.driver.finished or m.driver.stalled)
        ]
        events = [
            m for m in active if m.state in (RobotState.FALLEN, RobotState.RECOVERING)
        ]

        if events and current not in events:
            candidate = events[0]
        elif current is not None:
            terminal = current.driver.finished or current.driver.stalled
            if terminal and self._terminal_since_s is None:
                self._terminal_since_s = elapsed_s
            within_grace = terminal and (
                elapsed_s - self._terminal_since_s < SHOT_TERMINAL_GRACE_S
            )
            within_shot = not terminal and elapsed_s - self._shot_started_s < SHOT_MIN_S
            candidate = current if within_grace or within_shot else swarm.focus()
        else:
            candidate = swarm.focus()

        if candidate.name != self._subject_name:
            self._subject_name = candidate.name
            self._shot_started_s = elapsed_s
            self._terminal_since_s = None
            self.camera.reset()
        elif not (candidate.driver.finished or candidate.driver.stalled):
            self._terminal_since_s = None
        return candidate

    @staticmethod
    def _camera_companions(data, swarm: Swarm, subject) -> np.ndarray:
        """Positions that can usefully influence the current composition."""
        visible = [
            np.asarray(data.xpos[m.body_id], dtype=np.float64)
            for m in swarm.members
            if m is subject
            or (
                m.state is not RobotState.WAITING
                and not (m.driver.finished or m.driver.stalled)
            )
        ]
        return np.asarray(visible, dtype=np.float64)

    def _traces(self, swarm: Swarm) -> list[PathTrace]:
        return [
            PathTrace(
                livery=member.assignment.spec.livery,
                route=member.assignment.route,
                trail=self.trails.of(index),
                rejected_route=self.rejected_routes.get(member.name),
            )
            for index, member in enumerate(swarm.members)
        ]

    def _readouts(
        self, data, swarm: Swarm, subject, elapsed_s: float
    ) -> list[RobotReadout]:
        response_label = None
        if self.collapse_event_s is not None:
            response_label = (
                "CHECKING ROUTE" if elapsed_s <= self.collapse_event_s + 3.5 else None
            )
        return [
            RobotReadout(
                name=member.name,
                livery=member.assignment.spec.livery,
                storey=self._storey(float(data.xpos[member.body_id][2])),
                goal_storey=member.assignment.goal_storey,
                progress=member.progress,
                status=member.status,
                goal_label=member.assignment.goal_label,
                is_subject=member is subject,
                behaviour_label=(
                    (
                        response_label
                        or (
                            "SAFE ROUTE RETAINED"
                            if member.name in self.retained_robot_names
                            else "FOLLOWING NEW ROUTE"
                        )
                    )
                    if member.status == "driving" and self.collapse_event_s is not None
                    else None
                ),
            )
            for member in swarm.members
        ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cascade.demo", description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="use persistent robot GBP, geometry-aware radio "
        "and post-collapse graph rebuilds",
    )
    parser.add_argument(
        "--report", type=Path, help="write a machine-readable physical mission report"
    )
    parser.add_argument(
        "--robots", type=int, default=4, help="how many robots to deploy"
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="playback speed relative to real time (compact default: 1.2x)",
    )
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--no-render", action="store_true", help="skip video")
    parser.add_argument(
        "--no-cache", action="store_true", help="re-simulate the collapse"
    )
    parser.add_argument(
        "--live-disaster",
        action="store_true",
        help="render a bounded live-heat structural-failure mission",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="use the faster three-storey disaster preset with larger visible bays",
    )
    args = parser.parse_args(argv)
    if args.compact and not args.live_disaster:
        parser.error("--compact is a live-disaster preset; add --live-disaster")
    if args.distributed and not args.live_disaster:
        parser.error("--distributed requires --live-disaster")

    params = compact_demo_params() if args.compact else CascadeParams()
    if args.distributed:
        params = collision_aware_params(params, RobotSpec())
    out_dir = args.out / "compact" if args.compact else args.out
    experiment_config = {
        "params": asdict(params),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    before = capture_manifest(args.seed, experiment_config, {})

    result = run_demo(
        seed=args.seed,
        params=params,
        out_dir=out_dir,
        n_robots=args.robots,
        render=not args.no_render,
        use_cache=not args.no_cache,
        fps=args.fps,
        speed=(1.2 if args.compact else 2.0) if args.speed is None else args.speed,
        width=(960 if args.compact else 1120) if args.width is None else args.width,
        height=(865 if args.compact else 700) if args.height is None else args.height,
        live_disaster=args.live_disaster,
        distributed_physical=args.distributed,
    )
    print(result.report())
    report_path = args.report or (
        out_dir / f"seed{args.seed}_distributed.json" if args.distributed else None
    )
    if report_path is not None:
        metrics = asdict(result)
        metrics["storeys_reached"] = sorted(result.storeys_reached)
        metrics["planner_reachable"] = sorted(result.planner_reachable)
        metrics["video"] = str(result.video) if result.video else None
        manifest = capture_manifest(
            args.seed,
            experiment_config,
            {
                "accepted": result.accepted,
                "radio_bytes": result.radio_bytes,
                "distributed_physical": result.distributed_physical,
            },
        )
        if (before.git_sha, before.dirty_diff_hash) != (
            manifest.git_sha,
            manifest.dirty_diff_hash,
        ):
            raise RuntimeError(
                "source changed during physical validation; rerun from stable source"
            )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "metrics": metrics,
                    "accepted": result.accepted,
                    "provenance": asdict(manifest),
                    "physical_survivor_evacuation": False,
                    "embedded_timing_calibrated": False,
                    "whole_phase1_acceptance_gate_met": False,
                    "native_host_integer_routing": args.distributed,
                },
                indent=2,
            )
            + "\n"
        )
        if args.distributed:
            write_source_bundle(report_path.with_suffix(".source.tar.gz"))
    return 0 if result.accepted else 1


def _append_hold(
    writer: VideoWriter, frame: np.ndarray, *, fps: int, seconds: float
) -> None:
    """Append at least one copy of a terminal frame, then hold it for reading."""
    copies = max(1, round(fps * seconds))
    for _ in range(copies):
        writer.append(frame)


if __name__ == "__main__":
    raise SystemExit(main())
