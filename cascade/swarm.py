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
robot a substantial but bounded distance beyond that storey's closest safe
landing.  It is the cheapest thing that searches a *building* rather than a
floor, and it is a placeholder for the value-of-information gossip of SS6,
not a claim to be more than one.
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
from .robot import RobotSpec, RobotState, WaypointDriver, build_robot, place_robot

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
#: Final distance outside the facade before a robot drives square through the
#: opening.  Starting from a laterally offset queue position and aiming
#: directly at the first indoor graph node cuts the corner through the wall.
ENTRY_THRESHOLD_M: float = 0.45
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
#: Maximum extra graph cost spent sweeping a floor after reaching its nearest
#: certified landing.  Forty flat grid moves are about ten metres.  Without a
#: bound, the costliest-cell heuristic sent the storey-3 robot on a 262-node
#: tour through a second, physically fragile staircase even though the safe
#: stair used by the roof robot already reached that floor.
GOAL_FLOOR_SEARCH_BUDGET: float = 40.0
#: A destination is a search site, not a point in the raster margin outside
#: the facade.  Keep it far enough inside that the full chassis clears the
#: wall and its floating goal beacon unambiguously marks an interior site.
GOAL_INTERIOR_MARGIN_M: float = 1.50

#: A robot counts as settled when its chassis z-axis is at least this upright,
#: as a dot product with world up.  ~18 degrees of tilt.
SETTLED_UPRIGHT_DOT: float = 0.95
#: ...and when it is moving slower than this, in m/s and rad/s.
SETTLED_SPEED_M_S: float = 0.05
SETTLED_SPIN_RAD_S: float = 0.5
#: Minimum upward component of an external contact normal that can support a
#: robot.  This admits the 34-degree stair flights while excluding a chassis
#: that is merely touching a wall at the apex of a bounce.
SETTLED_SUPPORT_NORMAL_Z: float = 0.5
#: ...and has held both for this many consecutive steps.  A robot dropped onto
#: rubble bounces, and the instant of a bounce's apex satisfies the speed test
#: while the robot is very much still in the air.
SETTLED_HOLD_STEPS: int = 50


@dataclass(frozen=True, slots=True)
class Assignment:
    """Everything decided about one robot before the simulation starts."""

    spec: RobotSpec
    spawn: tuple[float, float, float]
    yaw_rad: float
    route: np.ndarray
    goal_storey: int
    goal_label: str = ""

    @property
    def name(self) -> str:
        return self.spec.name


@dataclass
class SwarmMember:
    """A robot under way, and what it has done so far."""

    assignment: Assignment
    driver: WaypointDriver
    body_id: int = -1
    #: Index of the root free joint's first velocity DOF, cached alongside
    #: `body_id` so the settling test does not re-look-up a joint per step.
    dof_adr: int = -1
    #: Every geom in the robot's free-joint subtree, for support-contact tests.
    geom_ids: frozenset[int] = field(default_factory=frozenset)
    visited: set[int] = field(default_factory=set)
    max_height_m: float = 0.0

    @property
    def name(self) -> str:
        return self.assignment.name

    @property
    def state(self) -> RobotState:
        return self.driver.state

    @property
    def status(self) -> str:
        return self.driver.state.value

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
    plan: FloorPlan, door_index: int, slot: int, spawn_z: float
) -> tuple[tuple[float, float, float], float]:
    """Where one robot starts: outside its exit, offset along the facade.

    `spawn_z` is the chassis centre height and comes from the platform's own
    geometry (`RobotSpec.spawn_height_m`).  It used to be a hardcoded 0.12 m,
    which was below the chassis's resting height and put every robot's wheels
    inside the ground.
    """
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
        pose = (x + outward * standoff, y + shift, spawn_z)
        yaw = math.pi if outward > 0 else 0.0
    else:
        outward = -1.0 if abs(door.at - footprint.y0) < 1e-6 else 1.0
        pose = (x + shift, y + outward * standoff, spawn_z)
        yaw = -0.5 * math.pi if outward > 0 else 0.5 * math.pi
    return pose, yaw


def _entry_approach(
    plan: FloorPlan,
    door_index: int,
    spawn: tuple[float, float, float],
    ride_height_m: float,
) -> np.ndarray:
    """Route from a queue slot to the doorway without crossing the facade.

    Robots first move parallel to the wall until they are centred on the
    opening, then approach it along the wall normal.  The old single diagonal
    from an offset spawn to an indoor node clipped the edge of the entrance.
    """

    door = plan.exits[door_index]
    footprint = plan.footprint
    surface_z = spawn[2] - ride_height_m
    start = np.asarray((spawn[0], spawn[1], surface_z), dtype=np.float64)
    x, y = door.center
    if door.axis.value == "x":
        outward = -1.0 if abs(door.at - footprint.x0) < 1e-6 else 1.0
        staged = np.asarray((spawn[0], y, surface_z))
        threshold = np.asarray((x + outward * ENTRY_THRESHOLD_M, y, surface_z))
    else:
        outward = -1.0 if abs(door.at - footprint.y0) < 1e-6 else 1.0
        staged = np.asarray((x, spawn[1], surface_z))
        threshold = np.asarray((x, y + outward * ENTRY_THRESHOLD_M, surface_z))
    points = [start]
    for point in (staged, threshold):
        if np.linalg.norm(point - points[-1]) > 0.05:
            points.append(point)
    return np.asarray(points)


def _goal_label(
    plan: FloorPlan,
    position: np.ndarray,
    storey: int,
) -> str:
    """Short semantic destination shown in the HUD."""

    if storey >= len(plan.storeys):
        return "ROOF · SEARCH"
    rooms = plan.storeys[storey].rooms
    room = next(
        (
            candidate
            for candidate in rooms
            if candidate.rect.contains_point(float(position[0]), float(position[1]))
        ),
        min(
            rooms,
            key=lambda candidate: (
                (candidate.rect.center[0] - position[0]) ** 2
                + (candidate.rect.center[1] - position[1]) ** 2
            ),
        ),
    )
    return f"S{storey} · {room.kind.value.upper()}-{room.bay}"


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
        np.abs(
            graph.surface_z
            - np.round(graph.surface_z / storey_height_m) * storey_height_m
        )
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

        # Search well beyond the landing, but do not select a remote cell whose
        # shortest path abandons the already certified vertical backbone for a
        # much longer staircase.  The graph cost is in flat-cell equivalents,
        # so this remains independent of the lattice's metric resolution.
        minimum = float(np.min(cost[pool]))
        budgeted = pool[cost[pool] <= minimum + GOAL_FLOOR_SEARCH_BUDGET]
        if budgeted.size:
            pool = budgeted

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
    poses = [
        _spawn_pose(plan, door, slot, spec.spawn_height_m) for door, slot in placements
    ]

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
    positions = graph.world_positions()
    footprint = plan.footprint
    interior = (
        (positions[:, 0] >= footprint.x0 + GOAL_INTERIOR_MARGIN_M)
        & (positions[:, 0] <= footprint.x1 - GOAL_INTERIOR_MARGIN_M)
        & (positions[:, 1] >= footprint.y0 + GOAL_INTERIOR_MARGIN_M)
        & (positions[:, 1] <= footprint.y1 - GOAL_INTERIOR_MARGIN_M)
    )
    best_cost[~interior] = np.inf
    goals = choose_goals(graph, best_cost, count, building.storey_height_m)

    assignments: list[Assignment] = []
    for index, ((door, _), (spawn, yaw)) in enumerate(
        zip(placements, poses, strict=True)
    ):
        start, dist, prev = solved[door]
        goal = goals[index % len(goals)] if goals else start
        route = path_from(dist, prev, start, goal)
        if route:
            graph_route = positions[route]
            approach = _entry_approach(
                plan,
                door,
                spawn,
                spec.ride_height_m,
            )
            waypoints = np.vstack((approach, graph_route))
        else:
            waypoints = np.zeros((0, 3))
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
                goal_label=(
                    _goal_label(plan, waypoints[-1], goal_storey)
                    if len(waypoints)
                    else "UNROUTABLE"
                ),
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

    def __init__(
        self,
        assignments: list[Assignment],
        *,
        departure_stagger_steps: int = DEPARTURE_STAGGER_STEPS,
        cruise_rad_s: float | None = None,
    ) -> None:
        if departure_stagger_steps < 0:
            raise ValueError("departure stagger must be non-negative")
        if cruise_rad_s is not None and cruise_rad_s <= 0.0:
            raise ValueError("cruise speed must be positive")
        driver_options = {} if cruise_rad_s is None else {"cruise_rad_s": cruise_rad_s}
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
                    start_delay_steps=index * departure_stagger_steps,
                    **driver_options,
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
            joint = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{item.name}_root"
            )
            if joint < 0:
                raise KeyError(f"robot {item.name!r} has no root free joint")
            member.dof_adr = int(model.jnt_dofadr[joint])
            member.geom_ids = frozenset(
                int(geom_id)
                for geom_id in range(model.ngeom)
                if int(model.body_rootid[model.geom_bodyid[geom_id]]) == member.body_id
            )

    def at_rest(self, data: mujoco.MjData) -> bool:
        """True when every robot is supported, upright, and slow."""
        for member in self.members:
            if member.body_id < 0 or member.dof_adr < 0:
                raise RuntimeError("place the swarm before checking whether it rests")
            rot = np.asarray(data.xmat[member.body_id], dtype=np.float64)
            if rot.reshape(3, 3)[2, 2] < SETTLED_UPRIGHT_DOT:
                return False
            velocity = data.qvel[member.dof_adr : member.dof_adr + 6]
            if float(np.linalg.norm(velocity[:3])) > SETTLED_SPEED_M_S:
                return False
            if float(np.linalg.norm(velocity[3:])) > SETTLED_SPIN_RAD_S:
                return False
            supported = False
            for contact in data.contact[: data.ncon]:
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
                first_is_robot = geom1 in member.geom_ids
                second_is_robot = geom2 in member.geom_ids
                if first_is_robot == second_is_robot:
                    continue
                # MuJoCo's contact normal points from geom1 to geom2.  Reverse
                # it when the robot is geom1 to get the normal acting on it.
                normal_z = float(contact.frame[2])
                if first_is_robot:
                    normal_z = -normal_z
                if normal_z >= SETTLED_SUPPORT_NORMAL_Z:
                    supported = True
                    break
            if not supported:
                return False
        return True

    def settle(
        self, model: mujoco.MjModel, data: mujoco.MjData, max_steps: int = 3000
    ) -> tuple[int, bool]:
        """Step until the swarm is resting.  Returns (steps used, settled).

        A fixed settling window is what this replaces, and it was wrong in both
        directions: too short for a robot dropped onto rubble outside a door,
        and pure waste for one that landed on flat ground in twenty steps.
        Worse, it was silent -- the drivers were switched on at step 250
        whether or not anything had landed, so a robot still in the air began
        its route mid-bounce.
        """
        if not self.members:
            return 0, True
        held = 0
        for step in range(1, max_steps + 1):
            mujoco.mj_step(model, data)
            if not self.at_rest(data):
                held = 0
                continue
            held += 1
            if held >= SETTLED_HOLD_STEPS:
                return step, True
        return max_steps, False

    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        for member in self.members:
            member.driver.step(model, data)

    def reroute(
        self,
        index: int,
        waypoints: np.ndarray,
        *,
        goal_label: str | None = None,
    ) -> None:
        """Install one robot's new route in control and presentation state."""

        if not 0 <= index < len(self.members):
            raise IndexError("swarm member index is out of range")
        member = self.members[index]
        member.driver.reroute(waypoints)
        member.assignment = dataclasses.replace(
            member.assignment,
            route=np.array(member.driver.waypoints, copy=True),
            goal_label=(
                member.assignment.goal_label if goal_label is None else goal_label
            ),
        )

    def observe(self, data: mujoco.MjData, storey_height_m: float) -> None:
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
        if not self.members:
            raise ValueError("an empty swarm has no camera focus")
        active = [
            m for m in self.members if not (m.driver.finished or m.driver.stalled)
        ]
        # A queued robot has zero progress by definition, so including it while
        # another robot is deployed makes the camera stare at the start line.
        # Assignment order breaks progress ties deterministically.
        deployed = [m for m in active if m.state is not RobotState.WAITING]
        return min(deployed or active or self.members, key=lambda m: m.progress)

    def positions(self, data: mujoco.MjData) -> np.ndarray:
        return np.array([data.xpos[m.body_id] for m in self.members], dtype=np.float64)

    @property
    def done(self) -> bool:
        return all(m.driver.finished or m.driver.stalled for m in self.members)

    @property
    def storeys_reached(self) -> set[int]:
        return {s for m in self.members for s in m.visited}
