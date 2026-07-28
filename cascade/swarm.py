"""A swarm of robots, and the assignment that spreads them through a ruin.

Phase 0's robots are teleoperated: each is handed a route by the same offline
planner and follows it open-loop.  That is deliberately *not* EMBER.  The point
of driving several at once is to establish that the world supports a swarm --
that there are several genuinely distinct places worth being, that the routes
to them are drivable, and that the scenario is legible when more than one
machine is moving in it.  The consensus-free routing that decides where each
robot should go, without a planner and without a map, is the whole rest of the
project.

Goal assignment here stratifies the reachable set by storey and sends each
robot to the costliest cell on one of them.  It is the cheapest thing that
searches a *building* rather than a floor, and it is a placeholder for the
value-of-information gossip of SS6, not a claim to be more than one.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from .extract import (
    TraversabilityGraph,
    dijkstra_from,
    nearest_node,
    path_from,
)
from .floorplan import FloorPlan
from .params import BuildingParams
from .robot import RobotSpec, WaypointDriver, build_robot, place_robot

#: Chassis colours, in assignment order.  Chosen to stay distinguishable
#: against grey concrete and from each other under the scene's dim lighting.
LIVERIES: tuple[tuple[float, float, float], ...] = (
    (0.95, 0.45, 0.10),  # orange
    (0.20, 0.75, 0.95),  # cyan
    (0.95, 0.25, 0.55),  # magenta
    (0.55, 0.85, 0.20),  # lime
    (0.98, 0.82, 0.15),  # yellow
    (0.65, 0.45, 0.95),  # violet
)

#: How far outside its exit a robot starts, metres.
SPAWN_STANDOFF_M: float = 1.2
#: Lateral spacing between robots queued at the same exit, metres.  Wider than
#: the chassis diagonal so they cannot spawn interpenetrating -- two bodies
#: sharing space resolve with an impulse large enough to put NaN in QACC on
#: the first step and kill the run.
SPAWN_SPACING_M: float = 0.85
#: Physics steps between one robot setting off and the next, at 0.002 s/step.
#: Robots do not collide with each other, so this is purely so the swarm reads
#: as a sequence of deployments on camera rather than four bodies leaving as
#: one indistinguishable clump.
DEPARTURE_STAGGER_STEPS: int = 1100
#: How far above or below a storey's floor level a cell may sit and still be
#: offered as a goal, metres.  Wide enough to include the top of a low debris
#: pile, narrow enough to exclude a stair flight.
GOAL_FLOOR_TOLERANCE_M: float = 0.20


@dataclass(frozen=True, slots=True)
class Assignment:
    """Everything decided about one robot before the simulation starts."""

    spec: RobotSpec
    spawn: tuple[float, float, float]
    yaw_rad: float
    route: np.ndarray
    goal_storey: int

    @property
    def name(self) -> str:
        return self.spec.name


@dataclass
class SwarmMember:
    """A robot under way, and what it has done so far."""

    assignment: Assignment
    driver: WaypointDriver
    body_id: int = -1
    visited: set[int] = field(default_factory=set)
    max_height_m: float = 0.0

    @property
    def name(self) -> str:
        return self.assignment.name

    @property
    def status(self) -> str:
        if self.driver.finished:
            return "arrived"
        if self.driver.stalled:
            return "stalled"
        return "driving"

    @property
    def progress(self) -> float:
        """Fraction of the route's waypoints passed, in [0, 1]."""
        total = len(self.assignment.route)
        if total == 0:
            return 1.0
        return min(1.0, self.driver.index / total)

    @property
    def reached_goal(self) -> bool:
        return self.assignment.goal_storey in self.visited


# --- assignment ---------------------------------------------------------------


def _entry_points(
    plan: FloorPlan, count: int, entry_doors: int
) -> list[tuple[int, int]]:
    """Assign `count` robots to exits, as (exit, slot) pairs.

    One door by default.  Splitting a small swarm across facades on opposite
    sides of the building makes it impossible to frame: the camera has to pull
    back far enough to hold both groups, and at that distance a 0.34 m robot is
    two pixels.  Robots that go in together and fan out inside are both easier
    to watch and a better picture of what the swarm is for.
    """
    usable = max(1, min(entry_doors, len(plan.exits)))
    slots: list[int] = [0] * usable
    out: list[tuple[int, int]] = []
    for index in range(count):
        door = index % usable
        out.append((door, slots[door]))
        slots[door] += 1
    return out


def _spawn_pose(
    plan: FloorPlan, door_index: int, slot: int
) -> tuple[tuple[float, float, float], float]:
    """Where one robot starts: outside its exit, offset along the facade."""
    door = plan.exits[door_index]
    footprint = plan.footprint
    x, y = door.center
    # Alternate left and right of the door centre so a queue stays compact and
    # symmetric rather than trailing off in one direction.
    rank = (slot + 1) // 2
    side = -1.0 if slot % 2 else 1.0
    shift = side * rank * SPAWN_SPACING_M
    # Robots further back in the queue also stand further out, so the one in
    # front is never blocking the doorway for the one behind.
    standoff = SPAWN_STANDOFF_M + rank * 0.55

    if door.axis.value == "x":
        outward = -1.0 if abs(door.at - footprint.x0) < 1e-6 else 1.0
        pose = (x + outward * standoff, y + shift, 0.12)
        yaw = math.pi if outward > 0 else 0.0
    else:
        outward = -1.0 if abs(door.at - footprint.y0) < 1e-6 else 1.0
        pose = (x + shift, y + outward * standoff, 0.12)
        yaw = -0.5 * math.pi if outward > 0 else 0.5 * math.pi
    return pose, yaw


def choose_goals(
    graph: TraversabilityGraph,
    cost: np.ndarray,
    count: int,
    storey_height_m: float,
) -> list[int]:
    """Pick `count` goals, one storey at a time, deepest-first.

    Stratifying by storey rather than sampling the reachable set directly is
    the difference between a swarm that searches a building and one that
    searches a floor.  Unstratified farthest-point sampling puts most goals on
    the ground storey for the simple reason that most reachable cells are
    there -- the ground floor is the only one with an outside -- and a rescue
    demo in which three of four robots never leave the entrance level is not
    showing the thing it exists to show.

    Within a storey the goal is the node that costs the most to reach, so
    robots are sent to the far end of the floor rather than just inside it,
    and subsequent goals on the same storey are pushed away from the ones
    already taken.
    """
    # Goals are on floors, never on stairs.  Unconstrained, "the costliest
    # reachable cell on storey 3" is reliably a cell most of the way up the
    # flight *to* storey 3 -- that is what costs the most to reach -- and the
    # demo then measured whether a robot could stop dead on a 34 degree ramp
    # one waypoint short of the landing.  All four robots drove their whole
    # route and then slid back down the last flight.  A place worth searching
    # is somewhere a casualty could be lying, which is a floor.
    on_floor = (
        np.abs(graph.surface_z - np.round(graph.surface_z / storey_height_m)
               * storey_height_m)
        <= GOAL_FLOOR_TOLERANCE_M
    )
    reachable = np.isfinite(cost) & on_floor
    if not reachable.any():
        return []

    positions = graph.world_positions()
    storey_of = np.floor(graph.surface_z / storey_height_m + 0.25).astype(int)
    # Highest first: the interesting claim is that upper storeys are reachable
    # at all, so the swarm should commit to them before it fills in the ground.
    order = sorted({int(s) for s in storey_of[reachable]}, reverse=True)
    if not order:
        return []

    chosen: list[int] = []
    taken_per_storey: dict[int, list[int]] = {}
    # A node may only be claimed once.  Without this a storey holding a single
    # reachable cell hands the same goal to every robot in turn, and the swarm
    # converges on one spot while reporting that it spread out.
    available = reachable.copy()

    for index in range(count):
        pool = np.zeros(0, dtype=np.int64)
        storey = order[0]
        # Round-robin over storeys, falling through to the next one when the
        # preferred storey has nothing left unclaimed.
        for offset in range(len(order)):
            storey = order[(index + offset) % len(order)]
            pool = np.flatnonzero(available & (storey_of == storey))
            if pool.size:
                break
        if pool.size == 0:
            break

        taken = taken_per_storey.setdefault(storey, [])
        if taken:
            # Push away from goals already claimed on this storey.
            separation = np.min(
                np.linalg.norm(
                    positions[pool][:, None, :] - positions[taken][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            pick = int(pool[int(np.argmax(separation))])
        else:
            pick = int(pool[int(np.argmax(cost[pool]))])

        available[pick] = False
        taken.append(pick)
        chosen.append(pick)
    return chosen


def plan_swarm(
    plan: FloorPlan,
    graph: TraversabilityGraph,
    building: BuildingParams,
    count: int,
    spec: RobotSpec | None = None,
    entry_doors: int = 1,
) -> list[Assignment]:
    """Spawn poses, goals and routes for a swarm of `count` robots."""
    spec = spec or RobotSpec()
    placements = _entry_points(plan, count, entry_doors)
    poses = [_spawn_pose(plan, door, slot) for door, slot in placements]

    # One Dijkstra per exit, not per robot: the sweep dominates planning and
    # every robot coming through the same door shares its answer.
    doors = sorted({door for door, _ in placements})
    solved = {}
    for door in doors:
        entry = plan.exits[door]
        node = nearest_node(graph, (entry.center[0], entry.center[1], 0.05))
        solved[door] = (node, *dijkstra_from(graph, node))

    # Goals are chosen over everything reachable from any exit, at the cheapest
    # cost from any of them, so two robots entering by different doors are
    # still spread against each other rather than each picking their own.
    best_cost = np.full(graph.n_nodes, np.inf)
    for _, dist, _ in solved.values():
        best_cost = np.minimum(best_cost, dist)
    goals = choose_goals(graph, best_cost, count, building.storey_height_m)

    assignments: list[Assignment] = []
    for index, ((door, _), (spawn, yaw)) in enumerate(
        zip(placements, poses, strict=True)
    ):
        start, dist, prev = solved[door]
        goal = goals[index % len(goals)] if goals else start
        route = path_from(dist, prev, start, goal)
        waypoints = (
            graph.world_positions()[route] if route else np.zeros((0, 3))
        )
        goal_storey = (
            math.floor(graph.surface_z[goal] / building.storey_height_m + 0.25)
            if route
            else 0
        )
        assignments.append(
            Assignment(
                spec=dataclasses.replace(
                    spec,
                    name=f"robot{index}",
                    livery=LIVERIES[index % len(LIVERIES)],
                ),
                spawn=spawn,
                yaw_rad=yaw,
                route=waypoints,
                goal_storey=goal_storey,
            )
        )
    return assignments


# --- the swarm under way ------------------------------------------------------


def swarm_bodies(
    assignments: list[Assignment],
) -> tuple[list[ET.Element], list[ET.Element]]:
    """MJCF worldbody and actuator elements for every robot."""
    bodies: list[ET.Element] = []
    actuators: list[ET.Element] = []
    for item in assignments:
        body, acts = build_robot(item.spec, item.spawn)
        bodies.append(body)
        actuators.extend(acts)
    return bodies, actuators


class Swarm:
    """Drives every robot and records where each of them got to."""

    def __init__(self, assignments: list[Assignment]) -> None:
        self.members = [
            SwarmMember(
                assignment=item,
                driver=WaypointDriver(
                    spec=item.spec,
                    waypoints=item.route,
                    # Released one at a time.  Four robots let go together at
                    # the same door arrive at it together, wedge, and every one
                    # behind the blockage is scored as stalled on geometry that
                    # was never the problem.
                    start_delay_steps=index * DEPARTURE_STAGGER_STEPS,
                ),
            )
            for index, item in enumerate(assignments)
        ]

    def place(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Teleport every robot to its spawn and cache its body id."""
        for member in self.members:
            item = member.assignment
            place_robot(model, data, item.spec, item.spawn, item.yaw_rad)
            member.body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, item.name
            )
            if member.body_id < 0:
                raise KeyError(f"robot {item.name!r} is not in the model")

    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        for member in self.members:
            member.driver.step(model, data)

    def observe(
        self, data: mujoco.MjData, storey_height_m: float
    ) -> None:
        """Record the storey each robot is on, from its own trajectory."""
        for member in self.members:
            height = float(data.xpos[member.body_id][2])
            member.max_height_m = max(member.max_height_m, height)
            storey = math.floor(height / storey_height_m + 0.25)
            if storey >= 0:
                member.visited.add(storey)

    def focus(self) -> SwarmMember:
        """The robot the camera should be watching: whoever has furthest to go.

        Robots that have arrived or given up are not interesting to look at,
        and a shot that stays on one after it stops is a shot of a parked
        robot while three others are still climbing.
        """
        active = [
            m for m in self.members if not (m.driver.finished or m.driver.stalled)
        ]
        return min(active or self.members, key=lambda m: m.progress)

    def positions(self, data: mujoco.MjData) -> np.ndarray:
        return np.array(
            [data.xpos[m.body_id] for m in self.members], dtype=np.float64
        )

    @property
    def done(self) -> bool:
        return all(m.driver.finished or m.driver.stalled for m in self.members)

    @property
    def storeys_reached(self) -> set[int]:
        return {s for m in self.members for s in m.visited}
