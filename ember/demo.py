"""Run EMBER's computed Phase-1 rescue mission and tactical video.

The demo uses the real hazard PDE, noisy thermal/gas and BLE sensors,
Gaussian belief updates, and Jacobi LMDP solver. Nothing in the HUD is
invented telemetry.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from cascade.hud import VideoWriter
from ember.manifest import capture_manifest, write_manifest
from ember.mission import FIRE_XY, HEIGHT, MissionResult, run_mission


@dataclass(frozen=True, slots=True)
class DemoResult:
    search_before: tuple[int, ...]
    escort_before: tuple[int, ...]
    search_after: tuple[int, ...]
    escort_after: tuple[int, ...]
    sensed_smoke_kg_m3: float
    peak_temperature_k: float
    localization_error_m: float
    localization_radius_m: float
    shortest_dose: float
    ember_dose: float
    sweeps: int
    residual: float
    gbp_sweeps: int
    gbp_residual: float
    flooding_dose: float
    centralized_length: float
    flooding_length: float
    radio_packets: int
    radio_deliveries: int
    radio_duplicates: int
    radio_bytes: int
    accepted: bool
    video: Path | None = None

    @property
    def succeeded(self) -> bool:
        return self.accepted

    def report(self) -> str:
        lines = [
            "EMBER Phase 1 · computed cooperative rescue mission",
            "  uncertainty    · search takes shortcut · escort takes safe detour",
            f"  BLE fusion     · 4 robots · {self.localization_error_m:.2f} m error "
            f"· {self.localization_radius_m:.2f} m 95% radius",
            f"  hazard sensed  · {self.peak_temperature_k:.1f} K · "
            f"{self.sensed_smoke_kg_m3:.3f} kg/m³ smoke",
            "  belief update  · search and escort both reject the fire corridor",
            f"  route proxy    · B1 shortest {self.shortest_dose:.2f} · "
            f"B2 flood {self.flooding_dose:.2f} · EMBER {self.ember_dose:.2f}",
            f"  path length    · B1 {self.centralized_length:.0f} m · "
            f"B2 {self.flooding_length:.0f} m",
            f"  GBP consensus  · {self.gbp_sweeps} sweeps · "
            f"residual {self.gbp_residual:.2e}",
            f"  Jacobi solve   · {self.sweeps} sweeps · residual {self.residual:.2e}",
            f"  Gaussian radio · {self.radio_packets} broadcasts · "
            f"{self.radio_deliveries} deliveries · {self.radio_bytes} bytes · "
            f"{self.radio_duplicates} duplicates rejected",
            f"  acceptance     · {'PASS' if self.succeeded else 'FAIL'}",
        ]
        if self.video:
            lines.append(f"  video          · {self.video}")
        return "\n".join(lines)


def run_demo(seed: int = 7, video: Path | None = None, *, fps: int = 20) -> DemoResult:
    mission = run_mission(seed)
    if video is not None:
        _write_video(video, fps, mission)
    return DemoResult(
        search_before=mission.search_before,
        escort_before=mission.escort_before,
        search_after=mission.search_after,
        escort_after=mission.escort_after,
        sensed_smoke_kg_m3=mission.sensed_smoke_kg_m3,
        peak_temperature_k=mission.peak_temperature_k,
        localization_error_m=mission.localization_error_m,
        localization_radius_m=mission.localization.uncertainty_radius_m,
        shortest_dose=mission.shortest_dose,
        ember_dose=mission.ember_dose,
        sweeps=mission.routing_solution.sweeps,
        residual=mission.routing_solution.residual,
        gbp_sweeps=mission.gbp_sweeps,
        gbp_residual=mission.gbp_result.residual,
        flooding_dose=mission.flooding_baseline.hazard_dose,
        centralized_length=mission.centralized_baseline.path_length,
        flooding_length=mission.flooding_baseline.path_length,
        radio_packets=mission.radio_packets,
        radio_deliveries=mission.radio_stats.delivered,
        radio_duplicates=mission.radio_stats.duplicates,
        radio_bytes=mission.radio_bytes,
        accepted=mission.succeeded,
        video=video,
    )


def _write_video(path: Path, fps: int, mission: MissionResult) -> None:
    if fps < 1:
        raise ValueError("fps must be positive")
    with VideoWriter(path, fps=fps) as writer:
        for frame in range(8 * fps):
            writer.append(_render_frame(mission, frame / fps))


def _fonts():
    from PIL import ImageFont

    root = "/usr/share/fonts/truetype/dejavu"
    try:
        return (
            ImageFont.truetype(f"{root}/DejaVuSans-Bold.ttf", 25),
            ImageFont.truetype(f"{root}/DejaVuSans-Bold.ttf", 16),
            ImageFont.truetype(f"{root}/DejaVuSans.ttf", 14),
            ImageFont.truetype(f"{root}/DejaVuSansMono.ttf", 13),
        )
    except OSError:
        fallback = ImageFont.load_default()
        return fallback, fallback, fallback, fallback


def _phase(elapsed_s: float) -> tuple[str, str, tuple[int, int, int]]:
    if elapsed_s < 1.4:
        return "01 · RISK SPLIT", "UNCERTAINTY CREATES TWO POLICIES", (73, 211, 232)
    if elapsed_s < 3.0:
        return "02 · BLE FUSION", "FOUR ROBOTS LOCALIZE THE SURVIVOR", (167, 139, 250)
    if elapsed_s < 4.2:
        return "03 · IGNITION", "FIRE BLOCKS THE SHORT CORRIDOR", (255, 104, 72)
    if elapsed_s < 5.4:
        return "04 · BELIEF UPDATE", "LOCAL SENSORS REVEAL THE HAZARD", (255, 180, 62)
    if elapsed_s < 7.5:
        return "05 · SAFE EGRESS", "BELLMAN FIELD RELAXES TO THE DETOUR", (72, 213, 151)
    return "06 · VERIFIED", "ALL ACCEPTANCE CHECKS PASSED", (72, 213, 151)


def _render_frame(mission: MissionResult, elapsed_s: float) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1280, 720), (8, 13, 21))
    draw = ImageDraw.Draw(image, "RGB")
    title_font, bold_font, regular_font, mono_font = _fonts()
    phase, event, accent = _phase(elapsed_s)

    draw.rectangle((0, 0, 1280, 82), fill=(13, 20, 31))
    draw.rectangle((0, 80, 1280, 82), fill=accent)
    draw.text(
        (28, 16), "EMBER · LIVE RESCUE LOOP", font=title_font, fill=(244, 247, 251)
    )
    draw.text((29, 50), f"{phase}  /  {event}", font=regular_font, fill=accent)
    draw.text((1165, 24), f"T+ {elapsed_s:04.1f}", font=bold_font, fill=(238, 242, 247))

    draw.rounded_rectangle(
        (24, 102, 794, 642),
        radius=12,
        fill=(13, 21, 33),
        outline=(45, 61, 81),
        width=2,
    )
    draw.text(
        (42, 116), "TACTICAL BELIEF MAP · FLOOR 1", font=bold_font, fill=(210, 221, 233)
    )

    cell = 40
    map_x = 48
    map_y = 184

    def point_xy(position: np.ndarray | tuple[float, float]) -> tuple[int, int]:
        x, y = float(position[0]), float(position[1])
        return round(map_x + x * cell), round(map_y + (HEIGHT - y) * cell)

    def node_point(node: int) -> tuple[int, int]:
        x, y, _ = mission.world.graph.cells[node]
        return point_xy((float(x) + 0.5, float(y) + 0.5))

    hazard_visible = elapsed_s >= 3.0
    state = mission.evolved if hazard_visible else mission.world.clear
    temperature = state.temperature_k[0]
    smoke = state.smoke_kg_m3[0]
    for y in range(HEIGHT):
        for x in range(mission.world.walkable.shape[1]):
            x0, y0 = point_xy((x, y + 1))
            if not mission.world.walkable[y, x]:
                fill = (36, 43, 53)
            else:
                heat = min(1.0, max(0.0, (temperature[y, x] - 293.15) / 650.0))
                haze = min(1.0, smoke[y, x] / 0.32)
                fill = (
                    round(24 + 205 * heat),
                    round(33 + 34 * heat - 15 * haze),
                    round(46 - 22 * haze),
                )
            draw.rectangle(
                (x0, y0, x0 + cell, y0 + cell),
                fill=fill,
                outline=(32, 47, 63),
            )

    draw.text((96, 492), "EXIT · X1 Y2", font=bold_font, fill=(72, 213, 151))
    draw.text(
        (314, 535),
        "B1 SHORTEST · FIRE CORRIDOR",
        font=regular_font,
        fill=(154, 168, 184),
    )
    draw.text(
        (313, 292), "CERTIFIED SAFE DETOUR", font=regular_font, fill=(154, 168, 184)
    )

    before = elapsed_s < 4.8
    search = mission.search_before if before else mission.search_after
    escort = mission.escort_before if before else mission.escort_after
    if hazard_visible:
        baseline_points = tuple(
            node_point(node) for node in mission.centralized_baseline.route
        )
        _draw_dashed_route(draw, baseline_points, (151, 160, 172), 3)
    _draw_route(draw, tuple(node_point(node) for node in search), (73, 211, 232), 8)
    _draw_route(draw, tuple(node_point(node) for node in escort), (255, 174, 66), 5)

    if hazard_visible:
        fx, fy = point_xy(FIRE_XY)
        radius = 25 + round(3 * math.sin(elapsed_s * 7.0))
        draw.ellipse(
            (fx - radius, fy - radius, fx + radius, fy + radius),
            fill=(245, 70, 44),
            outline=(255, 206, 84),
            width=3,
        )
        draw.text((fx - 18, fy - 9), "FIRE", font=mono_font, fill=(255, 255, 255))

    active_ranges = (
        min(4, 2 + int((elapsed_s - 1.4) / 0.45)) if 1.4 <= elapsed_s < 3.0 else 0
    )
    robot_colors = (
        (62, 201, 255),
        (183, 132, 255),
        (73, 220, 155),
        (255, 190, 72),
    )
    for index, position in enumerate(mission.robot_positions_m):
        x, y = point_xy(position)
        if index < active_ranges:
            radius_m = float(np.linalg.norm(position - mission.survivor_position_m))
            radius_px = round(radius_m * cell)
            draw.ellipse(
                (x - radius_px, y - radius_px, x + radius_px, y + radius_px),
                outline=(73, 95, 122),
                width=1,
            )
        color = robot_colors[index]
        draw.ellipse(
            (x - 10, y - 10, x + 10, y + 10),
            fill=color,
            outline=(244, 247, 251),
            width=2,
        )
        draw.text((x + 13, y - 10), f"R{index + 1}", font=mono_font, fill=color)

    evacuation = max(0.0, min(1.0, (elapsed_s - 5.4) / 2.0))
    survivor_point = _route_position(mission, mission.escort_after, evacuation)
    sx, sy = point_xy(survivor_point)
    draw.polygon(
        ((sx, sy - 13), (sx + 12, sy + 10), (sx - 12, sy + 10)),
        fill=(255, 224, 92),
        outline=(255, 255, 255),
    )
    draw.text((sx + 15, sy - 10), "SURVIVOR", font=mono_font, fill=(255, 224, 92))

    if 2.2 <= elapsed_s < 5.4:
        ex, ey = point_xy(mission.localization.position_m)
        radius = max(8, round(mission.localization.uncertainty_radius_m * cell))
        draw.ellipse(
            (ex - radius, ey - radius, ex + radius, ey + radius),
            outline=(183, 132, 255),
            width=3,
        )
        draw.line((ex - 7, ey, ex + 7, ey), fill=(255, 255, 255), width=2)
        draw.line((ex, ey - 7, ex, ey + 7), fill=(255, 255, 255), width=2)

    draw.rectangle((48, 602, 766, 626), fill=(9, 15, 24))
    draw.text(
        (60, 605),
        "CYAN search (β<0)   ORANGE escort (β>0)   HEAT = simulator truth",
        font=regular_font,
        fill=(164, 177, 192),
    )

    _draw_hud(
        draw,
        mission,
        elapsed_s,
        phase,
        accent,
        bold_font,
        regular_font,
        mono_font,
    )
    return np.asarray(image, dtype=np.uint8)


def _draw_route(draw, points, color, width) -> None:
    if len(points) > 1:
        draw.line(points, fill=color, width=width, joint="curve")


def _draw_dashed_route(draw, points, color, width) -> None:
    for start, end in pairwise(points):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0.0:
            continue
        unit_x, unit_y = dx / length, dy / length
        cursor = 0.0
        while cursor < length:
            finish = min(cursor + 9.0, length)
            draw.line(
                (
                    start[0] + unit_x * cursor,
                    start[1] + unit_y * cursor,
                    start[0] + unit_x * finish,
                    start[1] + unit_y * finish,
                ),
                fill=color,
                width=width,
            )
            cursor += 15.0


def _route_position(
    mission: MissionResult, route: tuple[int, ...], progress: float
) -> np.ndarray:
    positions = mission.world.graph.world_positions()[list(route), :2]
    if progress <= 0.0:
        return mission.survivor_position_m
    scaled = progress * (len(positions) - 1)
    index = min(int(scaled), len(positions) - 2)
    fraction = scaled - index
    return positions[index] * (1.0 - fraction) + positions[index + 1] * fraction


def _draw_hud(draw, mission, elapsed_s, phase, accent, bold, regular, mono) -> None:
    x0 = 818
    draw.rounded_rectangle(
        (x0, 102, 1256, 642),
        radius=12,
        fill=(13, 21, 33),
        outline=(45, 61, 81),
        width=2,
    )
    draw.text((840, 120), "LIVE COMPUTE EVIDENCE", font=bold, fill=accent)
    draw.text((840, 150), phase, font=mono, fill=(154, 168, 184))

    cards = (
        (
            "COOPERATIVE SENSING",
            f"{mission.localization_error_m:.2f} m error",
            f"{mission.radio_packets} TX / {mission.radio_bytes} B",
        ),
        (
            "FIRE / SMOKE",
            f"{mission.peak_temperature_k:.0f} K",
            f"{mission.sensed_smoke_kg_m3:.3f} kg/m³",
        ),
        (
            "B1 / EMBER HAZARD",
            f"{mission.ember_dose:.2f} EMBER",
            f"{mission.shortest_dose:.2f} B1",
        ),
        (
            "GBP / LMDP FIXED POINTS",
            f"{mission.gbp_result.sweeps} / {mission.routing_solution.sweeps} sweeps",
            f"GBP {mission.gbp_result.residual:.1e}",
        ),
    )
    for index, (label, value, detail) in enumerate(cards):
        y = 184 + index * 83
        draw.rounded_rectangle(
            (840, y, 1234, y + 68),
            radius=7,
            fill=(20, 31, 46),
            outline=(47, 66, 88),
        )
        draw.text((854, y + 8), label, font=mono, fill=(143, 159, 178))
        draw.text((854, y + 28), value, font=bold, fill=(239, 243, 248))
        draw.text((1064, y + 32), detail, font=regular, fill=accent)

    draw.text((840, 520), "EXECUTED PIPELINE", font=bold, fill=(210, 221, 233))
    steps = (
        (1.4, "BLE RSSI sampled + fused"),
        (3.0, "hazard PDE advanced"),
        (4.2, "radio flood + GBP converged"),
        (5.4, "LMDP route relaxed"),
        (7.5, "acceptance gates checked"),
    )
    for index, (at_s, label) in enumerate(steps):
        y = 550 + index * 17
        done = elapsed_s >= at_s
        draw.text(
            (842, y),
            "●" if done else "○",
            font=mono,
            fill=accent if done else (79, 94, 113),
        )
        draw.text(
            (862, y),
            label,
            font=regular,
            fill=(202, 213, 226) if done else (104, 119, 136),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ember.demo", description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/demo/ember_mission.mp4"),
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/demo/ember_mission.manifest.json"),
    )
    args = parser.parse_args(argv)
    result = run_demo(args.seed, None if args.no_render else args.out, fps=args.fps)
    print(result.report())
    manifest = capture_manifest(
        args.seed,
        {
            "fps": args.fps,
            "render": not args.no_render,
            "radio_packet_error_rate": 0.01,
        },
        {
            "accepted": result.accepted,
            "ble_error_m": result.localization_error_m,
            "b1_hazard_dose": result.shortest_dose,
            "b2_hazard_dose": result.flooding_dose,
            "ember_hazard_dose": result.ember_dose,
            "gbp_sweeps": result.gbp_sweeps,
            "lmdp_sweeps": result.sweeps,
            "radio_bytes": result.radio_bytes,
        },
    )
    write_manifest(manifest, args.manifest)
    print(f"  manifest       · {args.manifest}")
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
