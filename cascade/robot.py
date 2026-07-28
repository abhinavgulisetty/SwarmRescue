"""A small tracked-class ground robot, and a waypoint controller for it.

The robot is deliberately modest: a chassis and four driven wheels with high
friction, sized to fit through a 0.9 m doorway and to climb the building's
stairs.  Nothing here is EMBER -- this is the teleoperated platform Phase 0
uses to prove the world is navigable.  The onboard intelligence of SS5 arrives
in Phase 1 and runs against the same body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

#: Height of the visibility beacon above the chassis, metres.  Tall enough to
#: clear a 0.9 m partition so a robot in a room is still locatable from
#: outside it, short enough not to foul the storey above.
BEACON_HEIGHT_M: float = 1.10

#: Radius of the identification disc on top of the beacon, metres.  Sized for
#: the top-down pane: at 0.35 m it is about eight pixels across from the plan
#: camera's 28 m, which is the smallest a colour reads reliably at.
MARKER_RADIUS_M: float = 0.35

#: Collision masks.  Two geoms interact when `contype & conaffinity` is
#: non-zero in either direction, so putting robots on their own contype bit
#: while leaving their affinity on the world's bit means robots collide with
#: the building and the ground but pass through *each other*.
#:
#: This is a deliberate Phase 0 simplification and it needs stating plainly:
#: mutual collision avoidance is not modelled, so this demo does not
#: demonstrate it.  With contacts on, a swarm sharing one doorway simply piles
#: into it -- every robot wedges the one behind, all of them are scored as
#: stalled, and the run measures queueing at a door rather than whether the
#: ruin is navigable, which is the only question Phase 0 asks.  Deconfliction
#: is a routing-policy problem and belongs with the routing policy, in Phase 1.
_ROBOT_CONTYPE: str = "2"
_ROBOT_CONAFFINITY: str = "1"


@dataclass(frozen=True, slots=True)
class RobotSpec:
    """Physical parameters of the platform."""

    name: str = "robot"
    body_length_m: float = 0.34
    body_width_m: float = 0.24
    body_height_m: float = 0.10
    #: Wheel radius, metres.  Sized against the tallest unavoidable step on
    #: the route -- the lip at the foot of a stair flight, and the few
    #: centimetres a landing settles by -- rather than chosen for looks.  A
    #: rigid four-wheeler mounts a step of roughly its wheel radius driven
    #: square at it and rather less at an angle, so this is kept at or above
    #: `TraversabilityParams.max_step_height_m`: any less and the planner is
    #: certifying moves the chassis cannot make.
    wheel_radius_m: float = 0.13
    wheel_width_m: float = 0.045
    chassis_density: float = 700.0
    #: Wheel angular velocity limit, rad/s.
    max_wheel_rad_s: float = 22.0
    #: Servo gain on the wheel velocity actuators.
    wheel_gain: float = 12.0
    #: Sliding friction of the wheels.  High: the stairs are pitched at 34
    #: degrees and a wheel needs mu > tan(34) = 0.67 not to simply slip back
    #: down them.
    wheel_friction: float = 1.6
    #: Chassis colour.  Every robot in a swarm gets its own so that a still
    #: frame is readable: "the blue one is on storey 2" is a sentence you can
    #: only say if the blue one is distinguishable.
    livery: tuple[float, float, float] = (0.90, 0.45, 0.10)

    @property
    def ride_height_m(self) -> float:
        """Chassis centre height above the surface the wheels rest on."""
        return self.wheel_radius_m + 0.5 * self.body_height_m * 0.2


def build_robot(
    spec: RobotSpec, position: tuple[float, float, float]
) -> tuple[ET.Element, list[ET.Element]]:
    """Return the robot's worldbody element and its actuator elements."""
    half_l = 0.5 * spec.body_length_m
    half_w = 0.5 * spec.body_width_m
    half_h = 0.5 * spec.body_height_m

    body = ET.Element(
        "body",
        name=spec.name,
        pos=f"{position[0]} {position[1]} {position[2]}",
    )
    ET.SubElement(body, "freejoint", name=f"{spec.name}_root")
    ET.SubElement(
        body,
        "geom",
        name=f"{spec.name}_chassis",
        type="box",
        size=f"{half_l} {half_w} {half_h}",
        density=f"{spec.chassis_density}",
        rgba=f"{spec.livery[0]} {spec.livery[1]} {spec.livery[2]} 1",
        condim="3",
        contype=_ROBOT_CONTYPE,
        conaffinity=_ROBOT_CONAFFINITY,
    )
    # A marker mast, so the robot stays findable in a wide shot of a grey ruin.
    ET.SubElement(
        body,
        "geom",
        name=f"{spec.name}_mast",
        type="box",
        size="0.02 0.02 0.10",
        pos=f"0 0 {half_h + 0.10}",
        density="200",
        rgba=f"{spec.livery[0]} {spec.livery[1]} {spec.livery[2]} 1",
        condim="3",
        contype=_ROBOT_CONTYPE,
        conaffinity=_ROBOT_CONAFFINITY,
    )
    # A translucent beacon standing well above the chassis.  A 0.34 m robot in
    # a 24 x 18 x 12 m ruin is a few pixels in any shot wide enough to show
    # where it is, and the first cut of this demo was unwatchable for exactly
    # that reason -- you could not tell whether the robot was in frame at all.
    # Non-colliding and near-massless, so it is a label, not a part.
    ET.SubElement(
        body,
        "geom",
        name=f"{spec.name}_beacon",
        type="cylinder",
        size=f"0.018 {BEACON_HEIGHT_M * 0.5}",
        pos=f"0 0 {half_h + BEACON_HEIGHT_M * 0.5}",
        density="1",
        rgba=f"{spec.livery[0]} {spec.livery[1]} {spec.livery[2]} 0.55",
        contype="0",
        conaffinity="0",
    )
    # A disc on top of the beacon.  The mast and the beacon are both readable
    # from the side and both vanish from directly overhead, which is precisely
    # the view the plan pane takes -- a 0.34 m robot seen from 28 m up is two
    # pixels, and the pane whose whole job is to answer "where is the swarm"
    # answered it with four dots you had to hunt for.  Seen from above this is
    # a 0.7 m disc in the robot's own colour.
    ET.SubElement(
        body,
        "geom",
        name=f"{spec.name}_marker",
        type="cylinder",
        size=f"{MARKER_RADIUS_M} 0.012",
        pos=f"0 0 {half_h + BEACON_HEIGHT_M}",
        density="1",
        rgba=f"{spec.livery[0]} {spec.livery[1]} {spec.livery[2]} 0.85",
        contype="0",
        conaffinity="0",
    )

    actuators: list[ET.Element] = []
    wheel_x = half_l * 0.72
    wheel_y = half_w + 0.5 * spec.wheel_width_m
    for side, sy in (("l", 1.0), ("r", -1.0)):
        for row, sx in (("f", 1.0), ("b", -1.0)):
            wheel_name = f"{spec.name}_wheel_{side}{row}"
            hub = ET.SubElement(
                body,
                "body",
                name=wheel_name,
                pos=f"{sx * wheel_x} {sy * wheel_y} {-half_h * 0.2}",
            )
            ET.SubElement(
                hub,
                "joint",
                name=f"{wheel_name}_j",
                type="hinge",
                axis="0 1 0",
                damping="0.05",
            )
            ET.SubElement(
                hub,
                "geom",
                name=f"g_{wheel_name}",
                type="cylinder",
                size=f"{spec.wheel_radius_m} {0.5 * spec.wheel_width_m}",
                # A cylinder's axis is local z, so lay it on its side.
                euler="1.5707963 0 0",
                density="1200",
                rgba="0.15 0.15 0.17 1",
                friction=f"{spec.wheel_friction} 0.02 0.002",
                condim="3",
                contype=_ROBOT_CONTYPE,
                conaffinity=_ROBOT_CONAFFINITY,
            )
            actuators.append(
                ET.Element(
                    "velocity",
                    name=f"a_{wheel_name}",
                    joint=f"{wheel_name}_j",
                    kv=f"{spec.wheel_gain}",
                    ctrlrange=f"{-spec.max_wheel_rad_s} {spec.max_wheel_rad_s}",
                )
            )
    return body, actuators


#: Weight on height when projecting the robot onto its own route.  A route up
#: a staircase stacks waypoints almost vertically, so in plan distance the
#: cell at the top of a flight is a few centimetres from the cell at the
#: bottom.  Projecting in plan alone let a robot at the foot of the stairs
#: latch onto a waypoint above its head, aim at it, and grind into the riser
#: for the rest of the run.  Height has to count for more than plan distance,
#: not the same.
_HEIGHT_WEIGHT: float = 4.0


@dataclass
class WaypointDriver:
    """Differential-drive controller following a polyline.

    Deliberately simple: project onto the route, aim a little way beyond, slow
    down when badly misaligned, back out when wedged.  Phase 0 only needs to
    show the world is navigable; the routing policy that decides *where* to go
    is the subject of the whole rest of the project.
    """

    spec: RobotSpec
    waypoints: np.ndarray
    arrive_radius_m: float = 0.40
    #: Wheel speed on a straight run, rad/s.  With 0.13 m wheels this is
    #: about 0.5 m/s.  The first version cruised at 14 rad/s -- 1.05 m/s, a
    #: brisk walk -- which is far too fast to corner a 0.34 m chassis through a
    #: 0.9 m doorway that configuration-space planning leaves roughly 0.2 m of
    #: margin in.  It overshot every turn into the wall past it, rode up the
    #: skirting, and hung there.  The route was never the problem.
    cruise_rad_s: float = 4.0
    turn_gain: float = 7.0
    index: int = 0
    #: Set when the robot has stopped making progress toward its waypoint.
    stalled: bool = False
    #: Steps to hold still before setting off.  Robots share a doorway; a
    #: swarm released simultaneously piles into it and wedges, and every robot
    #: behind the blockage is scored as stalled through no fault of its own.
    start_delay_steps: int = 0
    #: Steps without advancing before declaring a stall.  Long -- 40 s of
    #: simulated time -- because with route re-acquisition a robot that has
    #: fallen off a staircase can genuinely drive back round to it, and the
    #: previous 18 s gave up on two robots that were within two waypoints of
    #: their goals.
    stall_patience: int = 12_000
    #: Steps without advancing before attempting to reverse out.
    recovery_after: int = 900
    #: Duration of one reverse-and-turn attempt.
    recovery_steps: int = 260
    #: Waypoints ahead to search when re-projecting onto the path.
    search_window: int = 12
    #: ...and behind, so that overshooting a turn is recoverable.  Short, so a
    #: switchback's other leg -- which passes within a metre in plan -- cannot
    #: capture the projection and send the robot back down the stairs.
    search_back: int = 6
    #: Distance from the projected waypoint past which the robot is treated as
    #: off its route entirely and re-acquires, metres.
    reacquire_m: float = 1.5
    #: How far back along the route re-acquisition may reach, in waypoints.
    #: About ten metres of route -- one flight of stairs and its landing.
    reacquire_back: int = 40
    #: Waypoints ahead of the projection to aim at.  Short: at a quarter-metre
    #: lattice, aiming 3 cells ahead lets the robot cut the inside of a corner
    #: into the wall it is turning around.
    lookahead: int = 2
    #: Floor on the cornering speed factor, so the robot still creeps through
    #: the sharpest turn rather than stopping dead in it.
    min_speed_factor: float = 0.22

    _steps: int = 0
    _best_index: int = -1
    _since_progress: int = 0
    _recovery_left: int = 0
    _recovery_turn: float = 1.0

    @property
    def finished(self) -> bool:
        return self.index >= len(self.waypoints)

    def target(self) -> np.ndarray:
        return self.waypoints[min(self.index, len(self.waypoints) - 1)]

    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._steps += 1
        if len(self.waypoints) == 0:
            self._drive(model, data, 0.0, 0.0)
            self.index = 0
            return
        # Arriving has to latch.  The projection below runs before the arrival
        # test, so a robot parked on its goal re-projected onto the last few
        # waypoints, found itself "not there yet", and set off again -- one
        # robot reached the roof, turned round, and drove back down two
        # storeys before the run ended.
        if self.finished or self.stalled:
            self._drive(model, data, 0.0, 0.0)
            return
        if self._steps <= self.start_delay_steps:
            self._drive(model, data, 0.0, 0.0)
            return

        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.spec.name
        )
        position = np.array(data.xpos[body_id], dtype=np.float64)
        rot = np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        forward = rot[:, 0]

        # Pure pursuit rather than "drive at waypoint k until within r".
        # Waypoints sit a voxel apart and the arrival radius is comparable, so
        # a robot that passes a waypoint slightly wide never registers arrival,
        # keeps chasing a point now behind it, and drives away down the
        # corridor it came from.  Projecting onto the nearest point *ahead* on
        # the path and aiming a fixed distance beyond it cannot get stuck that
        # way.
        # Waypoints are surface heights; `position` is the chassis centre, a
        # ride height above the surface.  Comparing them directly hands every
        # waypoint a constant height error, which the weight below multiplies
        # into a large one, and on a staircase that is enough to pin the
        # projection to a waypoint the robot has already passed.
        contact = position - np.array([0.0, 0.0, self.spec.ride_height_m])
        offset = self.waypoints - contact
        distance = np.sqrt(
            offset[:, 0] ** 2
            + offset[:, 1] ** 2
            + (_HEIGHT_WEIGHT * offset[:, 2]) ** 2
        )
        # The floor applies to both searches.  Bounding only the re-acquisition
        # leaves the short backward window free to walk the projection down the
        # route six waypoints at a time, which arrives at the same place a few
        # seconds later.
        floor = max(0, self._best_index - self.reacquire_back)
        window = slice(
            max(floor, self.index - self.search_back),
            min(self.index + self.search_window, len(distance)),
        )
        local = distance[window]
        if local.size:
            self.index = int(window.start + int(np.argmin(local)))
        # Lost.  A recovery manoeuvre can leave the robot metres from the
        # route -- backing off a stair ramp puts it under the flight it was
        # trying to climb -- and a projection that only ever moves forward has
        # no way back from that: it grinds toward a waypoint it can no longer
        # see for the rest of the run.  Re-acquiring the nearest point on the
        # route is what a teleoperator would do, and it is what turns a snag
        # into a detour instead of the end of the run.
        #
        # Bounded, though, and the bound is the whole difficulty.  A staircase
        # stacks its own waypoints four storeys deep in plan, so a robot that
        # slips near the roof is a metre from waypoints belonging to every
        # flight below it.  Unbounded, this re-acquisition sent two robots
        # that were three waypoints from the roof back down to the second
        # floor to start again.  Searching back a fixed distance along the
        # route recovers from a fall without ever undoing the climb.
        if distance[self.index] > self.reacquire_m:
            self.index = floor + int(np.argmin(distance[floor:]))

        if (
            self.index >= len(self.waypoints) - 1
            and distance[-1] <= self.arrive_radius_m
        ):
            self.index = len(self.waypoints)
            self._drive(model, data, 0.0, 0.0)
            return

        # One counter, reset only by real progress.  An earlier version also
        # zeroed it whenever a recovery began, which made the "no progress for
        # `stall_patience` steps" test unreachable and left a compound
        # condition that latched `stalled` on the first recovery attempt after
        # the patience expired -- so robots that backed out of a snag,
        # recovered and drove on for another twenty metres were still reported
        # as stalled, and the run stopped as soon as every robot had been
        # tarred once.
        if self.index > self._best_index:
            self._best_index = self.index
            self._since_progress = 0
        else:
            self._since_progress += 1

        if self._since_progress > self.stall_patience:
            self.stalled = True
            self._drive(model, data, 0.0, 0.0)
            return

        # Wedged: reverse while turning, then try again.  A skid-steer platform
        # on rubble gets caught on debris that it clears easily from a slightly
        # different approach, and without this the run ends at the first such
        # snag even though the route is perfectly drivable.  Attempts alternate
        # direction and repeat for as long as the patience lasts.
        if self._recovery_left > 0:
            self._recovery_left -= 1
            back = -0.45 * self.cruise_rad_s
            swing = 0.5 * self.turn_gain * self._recovery_turn
            self._drive(model, data, back - swing, back + swing)
            return
        if self._since_progress and self._since_progress % self.recovery_after == 0:
            self._recovery_left = self.recovery_steps
            self._recovery_turn = -self._recovery_turn
            return

        aim = min(self.index + self.lookahead, len(self.waypoints) - 1)
        delta = self.waypoints[aim] - position

        desired = math.atan2(delta[1], delta[0])
        heading = math.atan2(forward[1], forward[0])
        error = math.atan2(math.sin(desired - heading), math.cos(desired - heading))

        # Creep forward while turning hard rather than pivoting in place: a
        # four-wheel skid-steer platform turns badly when stationary on rubble.
        factor = max(self.min_speed_factor, math.cos(error))
        # Slow for the corner *before* reaching it.  Steering error alone only
        # reacts once the robot is already pointing the wrong way, which in a
        # corridor means once it is already committed into the wall.
        factor *= self._corner_factor(aim)
        speed = self.cruise_rad_s * factor
        turn = self.turn_gain * error
        self._drive(model, data, speed - turn, speed + turn)

    def _corner_factor(self, aim: int) -> float:
        """How much to slow for the bend just past the aim point, in [0, 1]."""
        beyond = min(aim + self.lookahead, len(self.waypoints) - 1)
        if beyond <= aim or aim <= self.index:
            return 1.0
        incoming = self.waypoints[aim, :2] - self.waypoints[self.index, :2]
        outgoing = self.waypoints[beyond, :2] - self.waypoints[aim, :2]
        length_in = float(np.linalg.norm(incoming))
        length_out = float(np.linalg.norm(outgoing))
        if length_in < 1e-6 or length_out < 1e-6:
            return 1.0
        alignment = float(
            np.clip(np.dot(incoming, outgoing) / (length_in * length_out), -1.0, 1.0)
        )
        # 1.0 dead ahead, min_speed_factor for a full reversal.
        return max(self.min_speed_factor, 0.5 * (1.0 + alignment))

    def _drive(
        self, model: mujoco.MjModel, data: mujoco.MjData, left: float, right: float
    ) -> None:
        limit = self.spec.max_wheel_rad_s
        for side, value in (("l", left), ("r", right)):
            for row in ("f", "b"):
                name = f"a_{self.spec.name}_wheel_{side}{row}"
                actuator = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
                )
                if actuator >= 0:
                    data.ctrl[actuator] = float(np.clip(value, -limit, limit))


def place_robot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: RobotSpec,
    position: tuple[float, float, float],
    yaw_rad: float = 0.0,
) -> None:
    """Teleport the robot to `position`, upright and at rest."""
    joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"{spec.name}_root"
    )
    if joint < 0:
        raise KeyError(f"no free joint for robot {spec.name!r}")
    adr = model.jnt_qposadr[joint]
    data.qpos[adr : adr + 3] = position
    data.qpos[adr + 3 : adr + 7] = (
        math.cos(0.5 * yaw_rad),
        0.0,
        0.0,
        math.sin(0.5 * yaw_rad),
    )
    dof = model.jnt_dofadr[joint]
    data.qvel[dof : dof + 6] = 0.0
    mujoco.mj_forward(model, data)
