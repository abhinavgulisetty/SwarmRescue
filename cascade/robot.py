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


@dataclass(frozen=True, slots=True)
class RobotSpec:
    """Physical parameters of the platform."""

    name: str = "robot"
    body_length_m: float = 0.34
    body_width_m: float = 0.24
    body_height_m: float = 0.10
    wheel_radius_m: float = 0.075
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
        rgba="0.90 0.45 0.10 1",
        condim="3",
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
        rgba="1.0 0.75 0.1 1",
        condim="3",
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


@dataclass
class WaypointDriver:
    """Differential-drive controller following a polyline.

    Deliberately simple: steer toward the next waypoint, slow down when badly
    misaligned, advance when close enough.  Phase 0 only needs to show the
    world is navigable; the routing policy that decides *where* to go is the
    subject of the whole rest of the project.
    """

    spec: RobotSpec
    waypoints: np.ndarray
    arrive_radius_m: float = 0.45
    cruise_rad_s: float = 14.0
    turn_gain: float = 6.0
    index: int = 0
    #: Set when the robot has stopped making progress toward its waypoint.
    stalled: bool = False
    _best_distance: float = float("inf")
    _stall_steps: int = 0
    #: Steps without closing on the target before declaring a stall.
    stall_patience: int = 5000
    #: Waypoints ahead to search when re-projecting onto the path.
    search_window: int = 12
    #: Waypoints ahead of the projection to aim at.
    lookahead: int = 3

    @property
    def finished(self) -> bool:
        return self.index >= len(self.waypoints)

    def target(self) -> np.ndarray:
        return self.waypoints[min(self.index, len(self.waypoints) - 1)]

    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.spec.name
        )
        position = np.array(data.xpos[body_id], dtype=np.float64)
        rot = np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        forward = rot[:, 0]

        if len(self.waypoints) == 0:
            self._drive(model, data, 0.0, 0.0)
            self.index = 0
            return

        # Pure pursuit rather than "drive at waypoint k until within r".
        # Waypoints sit 0.4 m apart on the voxel lattice and the arrival radius
        # is comparable, so a robot that passes a waypoint slightly wide never
        # registers arrival, keeps chasing a point now behind it, and drives
        # away down the corridor it came from.  Projecting onto the nearest
        # point *ahead* on the path and aiming a fixed distance beyond it
        # cannot get stuck that way.
        planar = np.hypot(
            self.waypoints[:, 0] - position[0], self.waypoints[:, 1] - position[1]
        )
        window = slice(self.index, min(self.index + self.search_window, len(planar)))
        local = planar[window]
        if local.size:
            self.index = int(self.index + int(np.argmin(local)))

        if self.index >= len(self.waypoints) - 1 and planar[-1] <= self.arrive_radius_m:
            self.index = len(self.waypoints)
            self._drive(model, data, 0.0, 0.0)
            return

        aim = min(self.index + self.lookahead, len(self.waypoints) - 1)
        delta = self.waypoints[aim] - position

        # Progress watchdog: a robot wedged against geometry the planner
        # thought was open should report that, not grind out the whole budget
        # and then be scored as "did not reach".
        remaining = float(len(self.waypoints) - self.index)
        if remaining < self._best_distance - 0.5:
            self._best_distance = remaining
            self._stall_steps = 0
        else:
            self._stall_steps += 1
            if self._stall_steps > self.stall_patience:
                self.stalled = True

        desired = math.atan2(delta[1], delta[0])
        heading = math.atan2(forward[1], forward[0])
        error = math.atan2(math.sin(desired - heading), math.cos(desired - heading))

        # Creep forward while turning hard rather than pivoting in place: a
        # four-wheel skid-steer platform turns badly when stationary on rubble.
        speed = self.cruise_rad_s * max(0.15, math.cos(error))
        turn = self.turn_gain * error
        self._drive(model, data, speed - turn, speed + turn)

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
