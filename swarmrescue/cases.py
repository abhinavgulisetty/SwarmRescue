"""Small, physically simulated situations for explaining the rescue workflow.

The authored layouts and timed debris releases are teaching setups. Wheel
control, visibility sensing, geometry extraction, integer routing and survivor
escort use the project runtime. No actor pose is overwritten after settling.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from cascade.cinema import _connect, _dashed_polyline, _polyline
from cascade.extract import build_graph, nearest_node, rasterize
from cascade.hud import _BODY_FONTS, VideoWriter, _font, story_frame
from cascade.params import TraversabilityParams
from cascade.render import CameraView, contact_sheet, render_scene, save_png
from cascade.robot import RobotSpec, WaypointDriver, build_robot
from ember.evacuation import EvacuationConfig, EvacuationTeam, survivor_bodies
from ember.geometry_sensing import box_raycasts
from ember.physical_team import ground_mission_graph
from ember.routing import build_lmdp
from mcu.native_log_lmdp import solve_native_log_q16

WIDTH_M, DEPTH_M = 7.5, 5.0
ROBOT_COLOUR = (0.12, 0.48, 0.88)
SURVIVOR_COLOUR = (0.95, 0.65, 0.16)


@dataclass(frozen=True)
class Case:
    name: str
    title: str
    question: str
    setup: str
    expected: str


CASES = {
    case.name: case
    for case in (
        Case(
            "contact",
            "01 / Find a survivor",
            "When does the robot actually find someone?",
            "A wall hides the survivor. The robot searches around it.",
            "local_contact",
        ),
        Case(
            "reroute",
            "02 / A passage becomes blocked",
            "What happens when the planned path is blocked?",
            "Debris is released above the direct passage, after the robot starts.",
            "destination_reached",
        ),
        Case(
            "escort",
            "03 / Escort someone outside",
            "Is reaching the survivor enough?",
            "One robot meets one survivor, then leads them through the doorway.",
            "survivor_evacuated",
        ),
        Case(
            "no-exit",
            "04 / There is no usable route",
            "What if the debris closes every passage?",
            "A wider falling member closes the entire route to the destination.",
            "safe_stop",
        ),
    )
}


def _box(world, name, pos, size, *, colour="0.58 0.61 0.63 1"):
    return ET.SubElement(
        world,
        "geom",
        name=name,
        type="box",
        pos=pos,
        size=size,
        rgba=colour,
        friction="1 0.02 0.005",
    )


class Situation:
    """One robot, a visible layout, and a small auditable event log."""

    def __init__(self, name):
        self.case = CASES[name]
        self.root = ET.Element("mujoco", model=f"case_{name}")
        ET.SubElement(self.root, "option", timestep="0.004", integrator="implicitfast")
        visual = ET.SubElement(self.root, "visual")
        ET.SubElement(visual, "global", offwidth="1280", offheight="720")
        ET.SubElement(
            visual, "headlight", diffuse="0.35 0.35 0.35", ambient="0.2 0.2 0.2"
        )
        world = ET.SubElement(self.root, "worldbody")
        ET.SubElement(
            world,
            "light",
            pos="3 2 8",
            dir="0 0 -1",
            directional="true",
            diffuse="0.45 0.45 0.45",
            ambient="0.1 0.1 0.1",
        )
        ET.SubElement(
            world,
            "geom",
            name="ground",
            type="plane",
            size="15 15 0.1",
            rgba="0.83 0.85 0.83 1",
            friction="1 0.02 0.005",
        )
        # Shallow walls leave the layout legible while retaining collision and
        # occlusion at the sensor height. The opening on the left is the exit.
        _box(world, "g_wall_south", "3.75 -0.12 0.5", "3.9 0.12 0.5")
        _box(world, "g_wall_north", "3.75 5.12 0.5", "3.9 0.12 0.5")
        _box(world, "g_wall_east", "7.62 2.5 0.5", "0.12 2.5 0.5")
        _box(world, "g_wall_west_lower", "-0.12 0.875 0.5", "0.12 0.875 0.5")
        _box(world, "g_wall_west_upper", "-0.12 4.125 0.5", "0.12 0.875 0.5")
        self.goal = np.array([3.75 if name == "escort" else 6.25, 2.5, 0.0])
        if name == "contact":
            _box(world, "g_wall_screen", "4.8 2.5 0.6", "0.12 0.8 0.6")
        self.block_id = None
        if name in ("reroute", "no-exit"):
            anchor = ET.SubElement(world, "body", name="anchor", pos="3.75 2.5 1.8")
            ET.SubElement(
                anchor,
                "geom",
                type="sphere",
                size="0.02",
                contype="0",
                conaffinity="0",
                rgba="0 0 0 0",
            )
            block = ET.SubElement(world, "body", name="debris", pos="3.75 2.5 1.8")
            ET.SubElement(block, "freejoint", name="debris_root")
            _box(
                block,
                "g_debris",
                "0 0 0",
                "0.35 2.5 0.35" if name == "no-exit" else "0.35 1.25 0.35",
                colour="0.78 0.31 0.22 1",
            )
            equality = ET.SubElement(self.root, "equality")
            ET.SubElement(
                equality, "weld", name="debris_support", body1="anchor", body2="debris"
            )
        self.spec = RobotSpec(name="robot0", livery=ROBOT_COLOUR)
        start = (
            (3.25, 2.5, self.spec.spawn_height_m)
            if name == "escort"
            else (3.0 if name == "contact" else 1.25, 2.5, self.spec.spawn_height_m)
        )
        body, actuators = build_robot(self.spec, start)
        # A short beacon keeps the robot recognizable without hiding its chassis.
        for geom in list(body.findall("geom")):
            if geom.attrib.get("name", "").endswith(("_beacon", "_marker")):
                body.remove(geom)
        if name == "escort":
            body.set("quat", "0 0 0 1")
        world.append(body)
        actuator = ET.SubElement(self.root, "actuator")
        actuator.extend(actuators)
        self.config = (
            EvacuationConfig(speed_m_s=0.65) if name == "escort" else EvacuationConfig()
        )
        if name in ("contact", "escort"):
            world.extend(survivor_bodies(np.array([self.goal]), self.config))
        if name == "escort":
            safe = _box(
                world,
                "safe_zone",
                "-0.95 2.5 0.003",
                "0.45 0.78 0.003",
                colour="0.45 0.73 0.55 1",
            )
            safe.set("contype", "0")
            safe.set("conaffinity", "0")
        self.model = mujoco.MjModel.from_xml_string(
            ET.tostring(self.root, encoding="unicode")
        )
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data, 150)
        self.robot_id = self.model.body("robot0").id
        if name in ("reroute", "no-exit"):
            self.block_id = self.model.geom("g_debris").id
        self.team = None
        if name in ("contact", "escort"):
            self.model.geom_rgba[self.model.geom("g_survivor0").id] = (
                *SURVIVOR_COLOUR,
                1,
            )
            self.team = EvacuationTeam(
                self.model,
                self.data,
                1,
                SimpleNamespace(x0=0, y0=0, x1=WIDTH_M, y1=DEPTH_M),
                config=self.config,
            )
        self.graph = self.extract_graph()
        self.route = self.plan(
            np.array([-1.0, 2.5, 0]) if name == "escort" else self.goal
        )
        if not len(self.route):
            raise RuntimeError("teaching setup has no initial route")
        self.old_route = np.empty((0, 3))
        self.driver = WaypointDriver(
            self.spec,
            self.route,
            arrive_radius_m=0.18,
            cruise_rad_s=4.0 if name == "reroute" else 6.0,
        )
        self.events = []
        self.event("search_started" if name == "contact" else "route_planned", 0)
        self.detected = False
        self.released = False
        self.replanned = False
        self.waiting = False
        self.finished_s = None
        self.outcome = None
        self.samples = []
        self.trail = []
        self.patient_trail = []
        self.rng = np.random.default_rng(42)
        self.hold_until_s = 0.8
        self.initial_robot = self.robot.copy()
        self.initial_patient = self.patient.copy() if self.team else None
        self.occluded_samples = 0

    @property
    def robot(self):
        return self.data.xpos[self.robot_id].copy()

    @property
    def patient(self):
        return self.team.positions(self.data)[0]

    def event(self, kind, elapsed_s, **details):
        self.events.append({"event": kind, "time_s": round(elapsed_s, 3), **details})

    def extract_graph(self):
        # Use an independent structural snapshot: actor bodies cannot become
        # obstacles in their own route graph or affect the physics model.
        root = ET.fromstring(ET.tostring(self.root))
        world = root.find("worldbody")
        for body in list(world.findall("body")):
            world.remove(body)
        for element in ("actuator", "equality"):
            found = root.find(element)
            if found is not None:
                root.remove(found)
        if self.block_id is not None:
            geom = self.block_id
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, self.data.geom_xmat[geom])
            debris = _box(
                world,
                "g_debris_snapshot",
                " ".join(map(str, self.data.geom_xpos[geom])),
                " ".join(map(str, self.model.geom_size[geom])),
            )
            debris.set("quat", " ".join(map(str, quat)))
        snapshot = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(snapshot)
        mujoco.mj_forward(snapshot, data)
        params = TraversabilityParams()
        grid = rasterize(snapshot, data, (WIDTH_M, DEPTH_M), 2.5, params)
        graph = build_graph(grid, params, snapshot, data)
        return ground_mission_graph(graph, maximum_height_m=0.1)

    def plan(self, goal):
        graph = self.graph
        target = nearest_node(graph, tuple(goal))
        origin = nearest_node(graph, tuple(self.robot))
        model = build_lmdp(graph, np.full(graph.n_nodes, 0.20), [target])
        result = solve_native_log_q16(model)
        if not result.fixed_point:
            raise RuntimeError("integer case planner did not converge")
        nodes = result.route(origin)
        if not nodes:
            return np.empty((0, 3))
        route = graph.world_positions()[list(nodes)]
        # Include the measured chassis contact point, not a pose correction.
        start = self.robot.copy()
        start[2] = 0
        return np.vstack((start, route))

    def stop(self):
        self.data.ctrl[:] = 0

    def step(self, elapsed_s):
        name = self.case.name
        if name in ("reroute", "no-exit"):
            if elapsed_s >= 2 and not self.released:
                self.data.eq_active[self.model.equality("debris_support").id] = 0
                self.released = True
                self.hold_until_s = 4.2
                self.event(
                    "debris_released",
                    elapsed_s,
                    trigger="authored timed support release",
                )
            if elapsed_s >= 4.2 and not self.replanned:
                self.graph = self.extract_graph()
                self.old_route = self.route.copy()
                self.route = self.plan(self.goal)
                self.replanned = True
                self.event(
                    "geometry_scanned",
                    elapsed_s,
                    debris_position_m=self.data.geom_xpos[self.block_id].tolist(),
                    graph_nodes=self.graph.n_nodes,
                )
                if len(self.route):
                    self.driver.reroute(self.route)
                    self.event("route_replaced", elapsed_s)
                else:
                    self.event("no_route", elapsed_s)
                    self.outcome = "safe_stop"
                    self.finished_s = elapsed_s
        if self.team and not self.team.contacted:
            measurement = self.team.sense(
                self.model, self.data, 0, self.robot, self.rng
            )
            if name == "contact":
                origin = self.robot + np.array([0, 0, 0.35])
                target = self.patient + np.array([0, 0, self.config.height_m / 2])
                hits, _ = box_raycasts(self.model, self.data, origin, target[None])
                self.occluded_samples += int(np.isfinite(hits[0]))
            if measurement is not None and not self.detected:
                self.detected = True
                self.event(
                    "survivor_detected",
                    elapsed_s,
                    measured_position_m=measurement.tolist(),
                )
            gap = float(np.linalg.norm(self.robot[:2] - self.patient[:2]))
            if measurement is not None and gap <= self.config.contact_radius_m:
                self.team.begin(self.data, 0, 0, self.robot, elapsed_s)
                self.event("local_contact", elapsed_s, distance_m=gap)
                if name == "contact":
                    self.finished_s, self.outcome = elapsed_s, "local_contact"
                else:
                    self.hold_until_s = elapsed_s + 2.0
        should_wait = False
        if self.team and name == "escort":
            should_wait = bool(self.team.control(self.data, self.robot[None])[0])
            gap = float(np.linalg.norm(self.robot[:2] - self.patient[:2]))
            # Resume after the survivor closes the gap, rather than flickering
            # at the threshold during the demonstrated pause.
            should_wait = should_wait or (self.waiting and gap > 0.55)
            if should_wait and not self.waiting:
                self.event(
                    "escort_waits",
                    elapsed_s,
                    gap_m=float(np.linalg.norm(self.robot[:2] - self.patient[:2])),
                )
            self.waiting = should_wait
        if (
            self.finished_s is not None
            or elapsed_s < self.hold_until_s
            or should_wait
            or not len(self.route)
        ):
            self.stop()
        else:
            self.driver.step(self.model, self.data)
        mujoco.mj_step(self.model, self.data)
        if self.team:
            self.team.observe(self.data, elapsed_s + self.model.opt.timestep)
            if self.team.done and self.finished_s is None:
                self.finished_s, self.outcome = elapsed_s, "survivor_evacuated"
                self.event(
                    "survivor_evacuated",
                    elapsed_s,
                    survivor_position_m=self.patient.tolist(),
                )
        if name == "reroute" and self.driver.finished and self.finished_s is None:
            self.finished_s, self.outcome = elapsed_s, "destination_reached"
            self.event("destination_reached", elapsed_s)

    def sample(self, elapsed_s):
        robot = self.robot
        point = robot.copy()
        point[2] = 0.02
        if not self.trail or np.linalg.norm(point - self.trail[-1]) > 0.07:
            self.trail.append(point)
        entry = {"time_s": round(elapsed_s, 3), "robot_m": robot.tolist()}
        if self.team:
            patient = self.patient
            entry["survivor_m"] = patient.tolist()
            entry["gap_m"] = float(np.linalg.norm(robot[:2] - patient[:2]))
            point = patient.copy()
            point[2] = 0.02
            if (
                not self.patient_trail
                or np.linalg.norm(point - self.patient_trail[-1]) > 0.07
            ):
                self.patient_trail.append(point)
        self.samples.append(entry)

    def caption(self, elapsed_s):
        name = self.case.name
        if self.outcome == "safe_stop":
            return "Every passage is blocked. The robot stops; no route is invented."
        if self.outcome == "destination_reached":
            return (
                "The robot reaches the destination by driving around the fallen debris."
            )
        if self.outcome == "survivor_evacuated":
            return (
                "The survivor has crossed the outdoor boundary. Evacuation is now "
                "complete."
            )
        if self.outcome == "local_contact":
            return (
                "Close-range contact is confirmed. Finding someone is a separate "
                "step from evacuation."
            )
        if name == "contact":
            return (
                "The survivor is now visible to the local sensor. The robot closes the "
                "remaining gap."
                if self.detected
                else (
                    "The wall hides the survivor from the robot. It follows the search "
                    "route around the wall."
                )
            )
        if name == "escort":
            if not self.team.contacted:
                return (
                    "The robot is close enough to establish contact before starting "
                    "the escort."
                )
            if elapsed_s < self.hold_until_s:
                return (
                    "Contact is confirmed. Next, both the robot and survivor must get "
                    "outside."
                )
            if self.waiting:
                return (
                    "The survivor falls behind. The robot pauses so the survivor can "
                    "catch up."
                )
            return "The survivor follows the robot's driven trail toward the doorway."
        if self.replanned:
            return (
                "The direct passage is blocked. The robot follows the newly "
                "calculated detour."
            )
        if self.released:
            return (
                "The support is released. The robot pauses while the debris falls "
                "and settles."
            )
        return "The direct passage is open. The robot starts along the original route."

    def frame(self, renderer, elapsed_s):
        def overlay(scene):
            if len(self.old_route):
                _dashed_polyline(scene, self.old_route, (0.90, 0.18, 0.12, 1), 0.025)
            if len(self.route):
                _polyline(scene, self.route, (0.17, 0.67, 0.43, 0.8), 0.018)
            _polyline(scene, np.array(self.trail), (*ROBOT_COLOUR, 1), 0.035)
            _polyline(scene, np.array(self.patient_trail), (*SURVIVOR_COLOUR, 1), 0.03)
            if self.case.name == "escort":
                _connect(
                    scene,
                    np.array([-0.5, 1.72, 0.04]),
                    np.array([-0.5, 3.28, 0.04]),
                    (0.12, 0.65, 0.36, 1),
                    0.035,
                )
            if self.case.name == "contact":
                origin = self.robot + np.array([0, 0, 0.35])
                target = self.patient + np.array([0, 0, self.config.height_m / 2])
                hit, _ = box_raycasts(self.model, self.data, origin, target[None])
                distance = float(np.linalg.norm(target - origin))
                end = origin + (target - origin) * min(1, hit[0] / max(distance, 1e-12))
                colour = (
                    (0.88, 0.20, 0.14, 1)
                    if np.isfinite(hit[0])
                    else (0.16, 0.67, 0.43, 1)
                )
                _connect(scene, origin, end, colour, 0.018)

        view = CameraView("case", (3.0, 2.5, 0), 7.0, 90, -67)
        frame = render_scene(
            self.model, self.data, view, renderer=renderer, overlay=overlay
        )
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        labels = [
            (self.robot + np.array([0, 0, 0.25]), "Robot", (22, 87, 156)),
            (np.array([0, 2.5, 0.05]), "Doorway", (40, 64, 53)),
        ]
        if self.case.name == "escort":
            labels[1] = (np.array([0, 3.25, 0.05]), "Doorway", (40, 64, 53))
            labels.append((np.array([-1.0, 1.9, 0.05]), "Safe zone", (36, 96, 55)))
        if self.team:
            labels.append(
                (
                    self.patient + np.array([0, 0, 0.85]),
                    "Survivor" if self.detected else "Survivor (viewer sees)",
                    (140, 84, 13),
                )
            )
        elif self.block_id is not None:
            labels.append(
                (
                    self.data.geom_xpos[self.block_id] + [0, 0, 0.4],
                    "Fallen debris" if self.replanned else "Suspended debris",
                    (146, 49, 31),
                )
            )
        if self.team is None:
            labels.append((self.goal, "Destination", (44, 76, 54)))
        camera = renderer.scene.camera[0]
        for point, label, colour in labels:
            pixel = _project(point, camera, frame.shape[1], frame.shape[0])
            if pixel is None:
                continue
            x, y = pixel
            font = _font(_BODY_FONTS, 17)
            length = draw.textlength(label, font=font)
            x = int(np.clip(x, 10, frame.shape[1] - length - 20))
            y = int(np.clip(y - 29, 8, frame.shape[0] - 35))
            draw.rounded_rectangle(
                (x - 6, y - 4, x + length + 6, y + 22), radius=4, fill=(249, 248, 244)
            )
            draw.text((x, y), label, font=font, fill=colour)
        legend = (
            "Blue: robot + driven trail   ·   Green: planned route   · Amber: survivor"
        )
        if self.block_id is not None:
            legend = (
                "Blue: robot + driven trail   ·   Green: current route   ·   Red "
                "dash: abandoned route"
            )
        return story_frame(
            np.asarray(image),
            title=self.case.title,
            situation=self.case.question,
            action=self.caption(elapsed_s),
            elapsed_s=elapsed_s,
            legend=legend,
            progress=(
                "Authored teaching layout · MuJoCo physics · simulation clock above"
            ),
        )


def _project(point, camera, width, height):
    relative = np.asarray(point) - camera.pos
    forward, up = np.asarray(camera.forward), np.asarray(camera.up)
    right = np.cross(forward, up)
    depth = float(relative @ forward)
    if depth <= 0:
        return None
    near = camera.frustum_near
    vertical = camera.frustum_top - camera.frustum_bottom
    horizontal = vertical * width / height
    x = width * (0.5 + float(relative @ right) * near / (depth * horizontal))
    y = height * (0.5 - float(relative @ up) * near / (depth * vertical))
    return x, y


def run_case(name, out, *, render=True, fps=15, speed=1.5, horizon_s=90):
    scene = Situation(name)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    screenshots = []
    next_frame, next_sample = 0.0, 0.0
    with ExitStack() as stack:
        renderer = None
        writer = None
        if render:
            renderer = mujoco.Renderer(scene.model, width=1280, height=518)
            stack.callback(renderer.close)
            writer = stack.enter_context(VideoWriter(out / f"{name}.mp4", fps=fps))
        for step in range(math.ceil(horizon_s / scene.model.opt.timestep)):
            elapsed_s = step * scene.model.opt.timestep
            scene.step(elapsed_s)
            if elapsed_s >= next_sample:
                scene.sample(elapsed_s)
                next_sample += 0.1
            if writer and elapsed_s >= next_frame:
                frame = scene.frame(renderer, elapsed_s)
                writer.append(frame)
                middle_event = {
                    "contact": "survivor_detected",
                    "reroute": "route_replaced",
                    "escort": "escort_waits",
                    "no-exit": "no_route",
                }[name]
                if not screenshots or (
                    len(screenshots) == 1
                    and any(e["event"] == middle_event for e in scene.events)
                ):
                    screenshots.append(frame)
                next_frame += speed / fps
            if scene.finished_s is not None and elapsed_s >= scene.finished_s + 3:
                break
        if writer:
            screenshots.append(scene.frame(renderer, elapsed_s))
            save_png(out / f"{name}-start.png", screenshots[0])
            save_png(out / f"{name}.png", screenshots[-1])
            save_png(
                out / f"{name}-storyboard.png", contact_sheet(screenshots, columns=1)
            )
    report = {
        "case": name,
        "title": scene.case.title,
        "question": scene.case.question,
        "setup": scene.case.setup,
        "expected_outcome": scene.case.expected,
        "outcome": scene.outcome or "timed_out",
        "accepted": scene.outcome == scene.case.expected,
        "duration_simulated_s": round(elapsed_s, 3),
        "playback_speed": speed,
        "video": f"{name}.mp4" if render else None,
        "thumbnail": f"{name}-start.png" if render else None,
        "events": scene.events,
        "robot_start_m": scene.initial_robot.tolist(),
        "robot_final_m": scene.robot.tolist(),
        "robot_travel_m": float(
            sum(
                np.linalg.norm(b - a)
                for a, b in zip(scene.trail, scene.trail[1:], strict=False)
            )
        ),
        "survivor_final_m": scene.patient.tolist() if scene.team else None,
        "survivor_travel_m": float(scene.team.distance_m[0]) if scene.team else None,
        "occluded_samples": scene.occluded_samples,
        "original_route_m": scene.old_route.tolist(),
        "final_route_m": scene.route.tolist(),
        "samples": scene.samples,
        "model_notes": [
            "The search site is authored; this case isolates detection and contact.",
            (
                "Authored small layout; timed support release in debris cases, not "
                "a heat-driven collapse."
            ),
            "Current full structural geometry is supplied to the integer planner.",
            (
                "Wheel-driven robot; force-driven upright survivor proxy; no gait "
                "or medical model."
            ),
            (
                "These cases explain components; they do not evaluate a "
                "distributed swarm or hardware deployment."
            ),
        ],
    }
    (out / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def write_gallery(out, reports):
    """A local, dependency-free player with chapters taken from actual events."""
    out = Path(out)
    data = json.dumps(reports).replace("<", "\\u003c")
    buttons = "\n".join(
        f'<button class="case" data-case="{i}">{html.escape(r["title"])}</button>'
        for i, r in enumerate(reports)
    )
    page = (Path(__file__).parent / "templates" / "cases.html").read_text()
    heading = "Rescue situations" if len(reports) > 1 else "Rescue situation"
    page = page.replace("GALLERY_TITLE", heading)
    page = page.replace("BUTTONS", buttons).replace("DATA", data)
    (out / "index.html").write_text(page)
    summary = {
        "accepted": all(r["accepted"] for r in reports),
        "cases": [
            {
                k: r[k]
                for k in (
                    "case",
                    "outcome",
                    "accepted",
                    "duration_simulated_s",
                    "events",
                )
            }
            for r in reports
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--out", type=Path, default=Path("outputs/cases"))
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--speed", type=float, default=1.5)
    args = parser.parse_args(argv)
    if (
        not 1 <= args.fps <= 60
        or not math.isfinite(args.speed)
        or not 0.25 <= args.speed <= 4
    ):
        parser.error("fps must be 1-60 and playback speed must be 0.25-4")
    names = list(CASES) if args.case == "all" else [args.case]
    reports = []
    for name in names:
        report = run_case(
            name, args.out, render=not args.no_render, fps=args.fps, speed=args.speed
        )
        reports.append(report)
        print(
            f"{name}: {report['outcome']} "
            f"({report['duration_simulated_s']} simulated seconds)",
            flush=True,
        )
    if not args.no_render:
        write_gallery(args.out, reports)
        print(args.out / "index.html")
    return 0 if all(r["accepted"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
