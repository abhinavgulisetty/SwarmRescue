"""Physical ambulatory-survivor proxies and a bounded escort lifecycle.

Survivors are upright, force-driven capsules with three translational degrees
of freedom. This idealizes balance and propulsion, rather than simulating gait
or medical physiology. Gravity, inertia and structural contacts are resolved by
MuJoCo. Their positions are never overwritten after scene initialization.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

from cascade.extract import dijkstra_from, nearest_node, path_from
from ember.geometry_sensing import box_raycasts


@dataclass(frozen=True, slots=True)
class EvacuationConfig:
    mass_kg: float = 55.0
    radius_m: float = 0.14
    capsule_half_length_m: float = 0.25
    speed_m_s: float = 0.65
    velocity_time_constant_s: float = 0.2
    maximum_acceleration_m_s2: float = 3.0
    contact_radius_m: float = 0.7
    sensor_radius_m: float = 2.0
    sensor_noise_m: float = 0.015
    maximum_escort_gap_m: float = 0.75
    trail_spacing_m: float = 0.10
    waypoint_radius_m: float = 0.12
    safe_zone_margin_m: float = 0.5
    maximum_trail_points: int = 2048
    sliding_friction: float = 0.08

    def __post_init__(self):
        for name in (
            "mass_kg",
            "radius_m",
            "capsule_half_length_m",
            "speed_m_s",
            "velocity_time_constant_s",
            "maximum_acceleration_m_s2",
            "contact_radius_m",
            "sensor_radius_m",
            "maximum_escort_gap_m",
            "trail_spacing_m",
            "waypoint_radius_m",
            "safe_zone_margin_m",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.sensor_noise_m) or self.sensor_noise_m < 0:
            raise ValueError("sensor noise must be finite and nonnegative")
        if not np.isfinite(self.sliding_friction) or self.sliding_friction < 0:
            raise ValueError("survivor friction must be finite and nonnegative")
        if self.sensor_radius_m < self.contact_radius_m:
            raise ValueError("sensor radius must cover contact radius")
        if (
            not isinstance(self.maximum_trail_points, int)
            or isinstance(self.maximum_trail_points, bool)
            or self.maximum_trail_points < 2
        ):
            raise ValueError("trail capacity must be at least two")

    @property
    def height_m(self):
        return 2 * (self.radius_m + self.capsule_half_length_m)


def survivor_bodies(sites, config=None):
    """Append physical survivor bodies without changing structural joint order."""
    cfg = config or EvacuationConfig()
    points = np.asarray(sites, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("survivor sites must be finite 3D points")
    bodies = []
    for index, point in enumerate(points):
        body = ET.Element(
            "body",
            name=f"survivor{index}",
            pos=" ".join(map(str, point + np.array([0, 0, 0.025]))),
        )
        for axis, direction in zip("xyz", ("1 0 0", "0 1 0", "0 0 1"), strict=True):
            ET.SubElement(
                body,
                "joint",
                name=f"survivor{index}_{axis}",
                type="slide",
                axis=direction,
                damping="0",
            )
        ET.SubElement(
            body,
            "geom",
            name=f"g_survivor{index}",
            type="capsule",
            size=f"{cfg.radius_m} {cfg.capsule_half_length_m}",
            pos=f"0 0 {cfg.height_m / 2}",
            mass=str(cfg.mass_kg),
            rgba="0.95 0.9 0.65 1",
            friction=f"{cfg.sliding_friction} 0.01 0.001",
            priority="1",
            contype="4",
            conaffinity="1",
            condim="3",
        )
        bodies.append(body)
    return bodies


class NoEvacuationRouteError(RuntimeError):
    """Current geometry disconnects the survivor from every outdoor zone."""


def exterior_route(graph, start, footprint, preferred=None):
    """Choose a reachable outdoor goal; a blocked preferred exit cannot win."""
    positions = graph.world_positions()
    depth = interior_depths(positions, footprint)
    origin = nearest_node(graph, tuple(start))
    distance, previous = dijkstra_from(graph, origin)
    candidates = np.flatnonzero(
        (depth <= -1.0) & (np.abs(positions[:, 2]) <= 0.1) & np.isfinite(distance)
    )
    if not candidates.size:
        raise NoEvacuationRouteError("no reachable outdoor evacuation zone")
    preference = (
        np.zeros(len(candidates))
        if preferred is None
        else np.linalg.norm(
            positions[candidates, :2] - np.asarray(preferred)[:2], axis=1
        )
    )
    order = np.lexsort((candidates, distance[candidates], preference))
    goal = int(candidates[order[0]])
    nodes = path_from(distance, previous, origin, goal)
    route = positions[nodes]
    if np.linalg.norm(route[0] - start) > 0.05:
        route = np.vstack((start, route))
    return positions[goal].copy(), route


def interior_depths(positions, footprint):
    points = np.asarray(positions)
    return np.minimum.reduce(
        (
            points[:, 0] - footprint.x0,
            footprint.x1 - points[:, 0],
            points[:, 1] - footprint.y0,
            footprint.y1 - points[:, 1],
        )
    )


class EvacuationTeam:
    """Unique local contact, breadcrumbs, physics actuation and verified egress."""

    def __init__(self, model, data, count, footprint, *, config=None):
        self.config = config or EvacuationConfig()
        self.footprint = footprint
        self.body_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"survivor{i}")
                for i in range(count)
            ]
        )
        if count < 1 or np.any(self.body_ids < 0):
            raise ValueError("physical survivor bodies are missing")
        joints = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"survivor{i}_x")
            for i in range(count)
        ]
        if any(joint < 0 for joint in joints):
            raise ValueError("survivor translation joints are missing")
        self.dof_addresses = model.jnt_dofadr[joints].copy()
        self.initial_positions = self.positions(data).copy()
        self.previous_positions = self.initial_positions.copy()
        self.distance_m = np.zeros(count)
        self.owner = np.full(count, -1, dtype=np.int32)
        self.contact_time_s = np.full(count, np.inf)
        self.evacuation_time_s = np.full(count, np.inf)
        self.trails = [deque() for _ in range(count)]
        self.last_passed_point = self.initial_positions.copy()
        self.trail_high_water = 0
        self.maximum_gap_m = 0.0

    def positions(self, data):
        return np.asarray(data.xpos[self.body_ids]).copy()

    @property
    def contacted(self):
        return int(np.isfinite(self.contact_time_s).sum())

    @property
    def rescued(self):
        return int(np.isfinite(self.evacuation_time_s).sum())

    @property
    def done(self):
        return self.rescued == len(self.body_ids)

    def sense(self, model, data, patient, robot_position, rng):
        """Local noisy position detection is gated by range and box visibility."""
        cfg = self.config
        point = self.positions(data)[patient]
        observer = np.asarray(robot_position, dtype=float) + np.array([0, 0, 0.35])
        target = point + np.array([0, 0, cfg.height_m / 2])
        if np.linalg.norm(target - observer) > cfg.sensor_radius_m:
            return None
        hit, _ = box_raycasts(model, data, observer, target[None, :])
        if np.isfinite(hit[0]):
            return None
        measured = point.copy()
        measured[:2] += rng.normal(0, cfg.sensor_noise_m, 2)
        return measured

    def begin(self, data, patient, robot, robot_position, elapsed_s):
        cfg = self.config
        position = np.asarray(robot_position, dtype=float)
        if (
            not 0 <= patient < len(self.body_ids)
            or not 0 <= robot < len(self.body_ids)
            or position.shape != (3,)
            or not np.all(np.isfinite(position))
            or not np.isfinite(elapsed_s)
            or elapsed_s < 0
        ):
            raise ValueError("contact identifiers, position and time must be valid")
        if self.owner[patient] >= 0:
            raise ValueError("survivor already has an escort")
        if robot in self.owner:
            raise ValueError("robot already owns a survivor")
        point = self.positions(data)[patient]
        if (
            np.linalg.norm(point[:2] - np.asarray(robot_position)[:2])
            > cfg.contact_radius_m
        ):
            raise ValueError("local contact has not been established")
        self.owner[patient] = robot
        self.contact_time_s[patient] = elapsed_s
        self.trails[patient].append(np.asarray(robot_position, dtype=float).copy())

    def control(self, data, robot_positions):
        """Apply bounded forces; return which escorts must wait for their patient."""
        cfg = self.config
        points = self.positions(data)
        wait = np.zeros(len(robot_positions), dtype=bool)
        for patient, owner in enumerate(self.owner):
            data.xfrc_applied[self.body_ids[patient]] = 0
            if owner < 0 or np.isfinite(self.evacuation_time_s[patient]):
                continue
            trail = self.trails[patient]
            robot = np.asarray(robot_positions[owner], dtype=float)
            gap = float(np.linalg.norm(robot[:2] - points[patient, :2]))
            self.maximum_gap_m = max(self.maximum_gap_m, gap)
            wait[owner] = gap >= cfg.maximum_escort_gap_m
            if np.linalg.norm(robot[:2] - trail[-1][:2]) >= cfg.trail_spacing_m:
                if len(trail) >= cfg.maximum_trail_points:
                    raise RuntimeError("escort breadcrumb capacity exhausted")
                trail.append(robot.copy())
            while len(trail) > 1:
                incoming = trail[0][:2] - self.last_passed_point[patient, :2]
                outgoing = trail[1][:2] - trail[0][:2]
                lengths = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
                turning = (
                    lengths > 1e-12 and float(incoming @ outgoing) / lengths < 0.985
                )
                radius = (
                    min(cfg.waypoint_radius_m, 0.035)
                    if turning
                    else cfg.waypoint_radius_m
                )
                if np.linalg.norm(trail[0][:2] - points[patient, :2]) > radius:
                    break
                self.last_passed_point[patient] = trail.popleft()
            self.trail_high_water = max(self.trail_high_water, len(trail))
            delta = trail[0][:2] - points[patient, :2]
            distance = float(np.linalg.norm(delta))
            # Intermediate breadcrumbs mark curvature, not stopping points.
            # Braking at each 10 cm sample makes the configured walking speed
            # unattainable even on a straight corridor. Slow only at the tail.
            speed = (
                cfg.speed_m_s if len(trail) > 1 else min(cfg.speed_m_s, 2 * distance)
            )
            desired = delta / max(distance, 1e-12) * speed
            adr = self.dof_addresses[patient]
            velocity = np.asarray(data.qvel[adr : adr + 2])
            acceleration = (desired - velocity) / cfg.velocity_time_constant_s
            norm = float(np.linalg.norm(acceleration))
            acceleration *= min(1.0, cfg.maximum_acceleration_m_s2 / max(norm, 1e-12))
            data.xfrc_applied[self.body_ids[patient], :2] = cfg.mass_kg * acceleration
        return wait

    def observe(self, data, elapsed_s):
        """Only physical survivor coordinates can establish rescue completion."""
        points = self.positions(data)
        self.distance_m += np.linalg.norm(points - self.previous_positions, axis=1)
        self.previous_positions[:] = points
        outdoors = (
            interior_depths(points, self.footprint) <= -self.config.safe_zone_margin_m
        )
        supported_height = np.abs(points[:, 2]) <= 0.25
        for patient in range(len(points)):
            if (
                self.owner[patient] >= 0
                and outdoors[patient]
                and supported_height[patient]
                and not np.isfinite(self.evacuation_time_s[patient])
            ):
                self.evacuation_time_s[patient] = elapsed_s
            elif self.owner[patient] >= 0 and (
                not outdoors[patient] or not supported_height[patient]
            ):
                self.evacuation_time_s[patient] = np.inf
