"""Continuous ground-route tracking with bounded wheel acceleration.

The original driver remains available for rubble and stair traversal. This
profile shares its lifecycle, but interpolates a lookahead by distance along
segments and damps turning with measured yaw velocity. Its path shortcuts are
checked against current structural boxes with a conservative robot envelope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .robot import WaypointDriver


def clear_segment(model, data, start, end, *, radius_m=0.31):
    """Conservative swept robot envelope against current oriented boxes.

    A horizontal disc and a 0.40 m tall vertical envelope grow each box in its
    local axes. A slab test then checks the entire segment, not isolated samples.
    Robot and survivor bodies are excluded, matching the simulation's actor masks.
    """
    origin = np.asarray(start, dtype=float) + np.array([0, 0, 0.20])
    target = np.asarray(end, dtype=float) + np.array([0, 0, 0.20])
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        name = model.geom(geom).name or ""
        if (
            "robot" in name
            or "survivor" in name
            or (model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0)
        ):
            continue
        rotation = data.geom_xmat[geom].reshape(3, 3)
        padding = radius_m * np.linalg.norm(rotation[:2], axis=0)
        padding += 0.20 * np.abs(rotation[2])
        half = model.geom_size[geom] + padding
        local = (origin - data.geom_xpos[geom]) @ rotation
        delta = (target - origin) @ rotation
        parallel = np.abs(delta) < 1e-12
        if np.any(parallel & (np.abs(local) > half)):
            continue
        first = np.divide(
            -half - local, delta, out=np.full(3, -np.inf), where=~parallel
        )
        second = np.divide(half - local, delta, out=np.full(3, np.inf), where=~parallel)
        enter, leave = (
            np.min([first, second], axis=0).max(),
            np.max([first, second], axis=0).min(),
        )
        if enter <= leave and leave >= 0 and enter <= 1:
            return False
    return True


def simplify_route(model, data, route, *, radius_m=0.38):
    """Remove lattice zigzags only when the swept shortcut is collision-free."""
    route = np.asarray(route, dtype=float)
    if len(route) < 3:
        return route.copy()
    selected = [route[0]]
    index = 0
    while index < len(route) - 1:
        # Try furthest first; all accepted shortcuts retain path order.
        for candidate in range(len(route) - 1, index, -1):
            if clear_segment(
                model, data, route[index], route[candidate], radius_m=radius_m
            ):
                index = candidate
                break
        else:
            raise ValueError("route has no collision-free next segment")
        selected.append(route[index])
    return np.array(selected)


@dataclass
class SmoothWaypointDriver(WaypointDriver):
    """Follow ground segments continuously rather than chasing grid vertices."""

    turn_gain: float = 3.5
    heading_damping: float = 1.2
    lookahead_m: float = 0.45
    acceleration_rad_s2: float = 3.0
    turning_acceleration_rad_s2: float = 5.0
    braking_rad_s2: float = 8.0
    recovery_after: int = 0
    _progress_m: float = field(default=0.0, init=False, repr=False)
    _linear_command: float = field(default=0.0, init=False, repr=False)
    _turn_command: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        for name in (
            "turn_gain",
            "lookahead_m",
            "acceleration_rad_s2",
            "turning_acceleration_rad_s2",
            "braking_rad_s2",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.heading_damping) or self.heading_damping < 0:
            raise ValueError("heading damping must be finite and nonnegative")

    def reroute(self, waypoints):
        changed = super().reroute(waypoints)
        if changed:
            self._progress_m = 0.0
        return changed

    def pause(self, model, data, *, emergency=False):
        if emergency:
            self._linear_command = self._turn_command = 0.0
        self._drive(model, data, 0, 0)

    def _steer(self, model, data, contact, position, forward):
        route = self.waypoints
        lengths = np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        projections = []
        for index, length in enumerate(lengths):
            if length < 1e-10 or cumulative[index] > self._progress_m + 1.0:
                continue
            span = route[index + 1, :2] - route[index, :2]
            fraction = float(
                np.clip((contact[:2] - route[index, :2]) @ span / length**2, 0, 1)
            )
            progress = cumulative[index] + fraction * length
            if progress < self._progress_m - 0.15:
                continue
            nearest = route[index, :2] + fraction * span
            projections.append((float(np.linalg.norm(nearest - contact[:2])), progress))
        if projections:
            self._progress_m = max(self._progress_m, min(projections)[1])
        remaining = max(0.0, cumulative[-1] - self._progress_m)
        if not len(lengths):
            remaining = float(np.linalg.norm(route[-1, :2] - contact[:2]))
        target = route[-1]
        visible = False
        for lookahead in (self.lookahead_m, 0.30, 0.20, 0.10):
            aim = min(cumulative[-1], self._progress_m + lookahead)
            segment = min(
                max(0, int(np.searchsorted(cumulative, aim) - 1)), len(lengths) - 1
            )
            if segment < 0:
                visible = clear_segment(model, data, contact, target, radius_m=0.30)
                break
            fraction = (aim - cumulative[segment]) / max(lengths[segment], 1e-12)
            target = route[segment] + fraction * (route[segment + 1] - route[segment])
            if clear_segment(model, data, contact, target, radius_m=0.30):
                visible = True
                break
        if not visible:
            self.pause(model, data)
            return
        delta = target[:2] - position[:2]
        heading = math.atan2(forward[1], forward[0])
        desired = math.atan2(delta[1], delta[0])
        error = math.atan2(math.sin(desired - heading), math.cos(desired - heading))
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            model.body(self.spec.name).id,
            velocity,
            0,
        )
        curvature = 2 * math.sin(error) / max(float(np.linalg.norm(delta)), 0.10)
        factor = max(0.0, math.cos(error)) ** 2
        factor /= 1 + 0.18 * abs(curvature)
        speed = self.cruise_rad_s * factor
        # Brake continuously at the end rather than hitting the goal at cruise.
        speed = min(
            speed,
            math.sqrt(
                2 * self.acceleration_rad_s2 * remaining / self.spec.wheel_radius_m
            ),
        )
        turn = np.clip(
            self.turn_gain * error - self.heading_damping * velocity[2], -5.0, 5.0
        )
        self._drive(model, data, speed - turn, speed + turn)

    def _drive(self, model, data, left, right):
        linear, turn = (left + right) / 2, (right - left) / 2
        dt = model.opt.timestep
        rate = (
            self.braking_rad_s2
            if abs(linear) < abs(self._linear_command)
            else self.acceleration_rad_s2
        )
        self._linear_command += float(
            np.clip(linear - self._linear_command, -rate * dt, rate * dt)
        )
        self._turn_command += float(
            np.clip(
                turn - self._turn_command,
                -self.turning_acceleration_rad_s2 * dt,
                self.turning_acceleration_rad_s2 * dt,
            )
        )
        super()._drive(
            model,
            data,
            self._linear_command - self._turn_command,
            self._linear_command + self._turn_command,
        )
