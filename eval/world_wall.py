"""Render a cinematic 20-world parallel replay from real EMBER evaluations."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from cascade.hud import VideoWriter
from eval.ablations import AblationConfig, _gbp_belief, _run_variant
from eval.harness import parse_seed_spec
from eval.procedural import (
    ProceduralEvaluationConfig,
    build_procedural_world,
    sample_exogenous_streams,
)

WIDTH = 1920
HEIGHT = 1080
HEADER_HEIGHT = 118
FOOTER_HEIGHT = 104
BACKGROUND = (5, 9, 18)
PANEL = (11, 18, 30)
PANEL_ACTIVE = (14, 24, 39)
RULE = (37, 55, 75)
TEXT = (235, 242, 249)
MUTED = (126, 148, 171)
CYAN = (63, 221, 244)
RED = (255, 91, 91)
AMBER = (255, 183, 77)
GREEN = (79, 220, 151)


@dataclass(frozen=True, slots=True)
class WorldReplay:
    seed: int
    positions: np.ndarray
    edges: np.ndarray
    exits: tuple[int, ...]
    survivors: tuple[int, ...]
    robots: tuple[int, ...]
    hazard: np.ndarray
    ember_routes: tuple[tuple[int, ...], ...]
    flooding_routes: tuple[tuple[int, ...], ...]
    ember_objective: float
    flooding_objective: float
    ember_bytes: int
    flooding_bytes: int
    inference_sweeps: int
    routing_sweeps: int

    @property
    def objective_improvement(self) -> float:
        return self.flooding_objective - self.ember_objective


@dataclass(frozen=True, slots=True)
class WallSummary:
    world_count: int
    survivor_count: int
    ember_wins: int
    ties: int
    flooding_wins: int
    mean_ember_objective: float
    mean_flooding_objective: float
    mean_ember_bytes: float
    mean_flooding_bytes: float

    @property
    def byte_reduction(self) -> float:
        return 1.0 - self.mean_ember_bytes / self.mean_flooding_bytes


def _fonts():
    from PIL import ImageFont

    root = "/usr/share/fonts/truetype/dejavu"
    try:
        return (
            ImageFont.truetype(f"{root}/DejaVuSans-Bold.ttf", 34),
            ImageFont.truetype(f"{root}/DejaVuSans-Bold.ttf", 19),
            ImageFont.truetype(f"{root}/DejaVuSans.ttf", 15),
            ImageFont.truetype(f"{root}/DejaVuSansMono-Bold.ttf", 14),
            ImageFont.truetype(f"{root}/DejaVuSansMono.ttf", 12),
        )
    except OSError:
        fallback = ImageFont.load_default()
        return fallback, fallback, fallback, fallback, fallback


def _project_positions(positions: np.ndarray) -> np.ndarray:
    """Project multi-storey xyz nodes into a legible miniature isometric view."""

    points = np.asarray(positions, dtype=np.float64)
    projected = np.column_stack(
        (
            points[:, 0] + 0.30 * points[:, 2],
            points[:, 1] - 0.46 * points[:, 2],
        )
    )
    low = np.min(projected, axis=0)
    span = np.maximum(np.ptp(projected, axis=0), 1.0)
    return (projected - low) / span


def _hazard_color(value: float, low: float, high: float) -> tuple[int, int, int]:
    level = float(np.clip((value - low) / max(high - low, 1.0e-12), 0.0, 1.0))
    if level < 0.55:
        blend = level / 0.55
        left = np.asarray((25, 54, 70))
        right = np.asarray((246, 173, 62))
    else:
        blend = (level - 0.55) / 0.45
        left = np.asarray((246, 173, 62))
        right = np.asarray((239, 57, 73))
    return tuple(
        int(item) for item in np.rint(left * (1.0 - blend) + right * blend)
    )


def _grid_shape(count: int) -> tuple[int, int]:
    if count < 1 or count > 20:
        raise ValueError("world wall supports between 1 and 20 worlds")
    columns = 5 if count > 12 else 4 if count > 6 else 3
    return columns, math.ceil(count / columns)


def _summary(replays: tuple[WorldReplay, ...]) -> WallSummary:
    differences = np.asarray(
        [replay.objective_improvement for replay in replays],
        dtype=np.float64,
    )
    tolerance = 1.0e-9
    return WallSummary(
        world_count=len(replays),
        survivor_count=sum(len(replay.survivors) for replay in replays),
        ember_wins=int(np.sum(differences > tolerance)),
        ties=int(np.sum(np.abs(differences) <= tolerance)),
        flooding_wins=int(np.sum(differences < -tolerance)),
        mean_ember_objective=float(
            np.mean([replay.ember_objective for replay in replays])
        ),
        mean_flooding_objective=float(
            np.mean([replay.flooding_objective for replay in replays])
        ),
        mean_ember_bytes=float(
            np.mean([replay.ember_bytes for replay in replays])
        ),
        mean_flooding_bytes=float(
            np.mean([replay.flooding_bytes for replay in replays])
        ),
    )


def build_replays(
    seeds: tuple[int, ...],
    *,
    noise_seed: int = 0,
) -> tuple[WorldReplay, ...]:
    """Build the same paired full/no-VoI states used by the ablation report."""

    ablation_config = AblationConfig(
        world_seeds=seeds,
        noise_seed=noise_seed,
    )
    world_config = ProceduralEvaluationConfig(
        world_seeds=seeds,
        noise_seeds=(noise_seed,),
        voi_packet_budget_bytes=ablation_config.voi_packet_budget_bytes,
        lmdp_control_temperature=ablation_config.control_temperature,
        uncertainty_beta=ablation_config.uncertainty_beta,
    )
    replays = []
    for seed in seeds:
        world = build_procedural_world(seed, world_config)
        streams = sample_exogenous_streams(
            world,
            noise_seed,
            voi_packet_budget_bytes=ablation_config.voi_packet_budget_bytes,
            lmdp_control_temperature=ablation_config.control_temperature,
        )
        ember_belief, ember_inference = _gbp_belief(
            world,
            streams.voi_received_packets,
        )
        flooding_belief, flooding_inference = _gbp_belief(
            world,
            streams.received_packets,
        )
        ember, _, _, ember_routes = _run_variant(
            world,
            "full",
            ember_belief,
            ember_inference,
            streams.voi_radio_stats.bytes_air,
            ablation_config,
        )
        flooding, _, _, flooding_routes = _run_variant(
            world,
            "no_voi",
            flooding_belief,
            flooding_inference,
            streams.radio_stats.bytes_air,
            ablation_config,
        )
        replays.append(
            WorldReplay(
                seed=seed,
                positions=_project_positions(world.graph.world_positions()),
                edges=np.asarray(world.graph.edges, dtype=np.int64),
                exits=world.exit_nodes,
                survivors=world.survivor_nodes,
                robots=world.robot_nodes,
                hazard=np.asarray(world.dynamic_cost, dtype=np.float32),
                ember_routes=ember_routes,
                flooding_routes=flooding_routes,
                ember_objective=ember.mean_route_objective,
                flooding_objective=flooding.mean_route_objective,
                ember_bytes=ember.radio_bytes,
                flooding_bytes=flooding.radio_bytes,
                inference_sweeps=ember.inference_sweeps,
                routing_sweeps=ember.routing_sweeps,
            )
        )
    return tuple(replays)


def _point(
    normalized: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> tuple[int, int]:
    left, top, right, bottom = bounds
    x = left + 10 + float(normalized[0]) * (right - left - 20)
    y = bottom - 9 - float(normalized[1]) * (bottom - top - 18)
    return round(x), round(y)


def _route_point(
    replay: WorldReplay,
    route: tuple[int, ...],
    progress: float,
    bounds: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    if not route:
        return None
    if len(route) == 1:
        return _point(replay.positions[route[0]], bounds)
    scaled = float(np.clip(progress, 0.0, 1.0)) * (len(route) - 1)
    segment = min(int(scaled), len(route) - 2)
    blend = scaled - segment
    location = (
        replay.positions[route[segment]] * (1.0 - blend)
        + replay.positions[route[segment + 1]] * blend
    )
    return _point(location, bounds)


def _draw_dashed(draw, points, fill, width: int) -> None:
    for index, (left, right) in enumerate(pairwise(points)):
        if index % 2 == 0:
            draw.line((left, right), fill=fill, width=width)


def _phase(elapsed_s: float) -> tuple[str, str, tuple[int, int, int]]:
    if elapsed_s < 1.2:
        return "BOOT", "SPAWNING 20 PROCEDURAL DISASTERS", CYAN
    if elapsed_s < 3.3:
        return "SENSE", "LOSSY RADIO · LOCAL EVIDENCE · 60 SURVIVORS", AMBER
    if elapsed_s < 6.0:
        return "INFER", "GBP + VALUE-OF-INFORMATION SCHEDULING", (176, 126, 255)
    if elapsed_s < 10.2:
        return "ROUTE", "ALL WORLDS RELAX TOWARD EGRESS", GREEN
    return "RESULT", "20-WORLD DEVELOPMENT GRID COMPLETE", CYAN


def render_frame(
    replays: tuple[WorldReplay, ...],
    elapsed_s: float,
    *,
    duration_s: float = 12.0,
) -> np.ndarray:
    """Render one synchronized dashboard frame."""

    from PIL import Image, ImageDraw

    if not replays:
        raise ValueError("at least one world replay is required")
    title_font, bold_font, _regular_font, mono_font, tiny_font = _fonts()
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image, "RGB")
    phase, phase_text, accent = _phase(elapsed_s)
    summary = _summary(replays)

    draw.rectangle((0, 0, WIDTH, HEADER_HEIGHT), fill=(9, 15, 26))
    draw.rectangle((0, HEADER_HEIGHT - 3, WIDTH, HEADER_HEIGHT), fill=accent)
    draw.text((32, 17), "EMBER // PARALLEL WORLD WALL", font=title_font, fill=TEXT)
    draw.text(
        (34, 65),
        f"{phase}  ·  {phase_text}",
        font=bold_font,
        fill=accent,
    )
    draw.text(
        (1534, 24),
        f"T+ {elapsed_s:04.1f}s",
        font=bold_font,
        fill=TEXT,
    )
    draw.text(
        (1534, 60),
        "DEV SEEDS 000-019 · REPLAY",
        font=mono_font,
        fill=MUTED,
    )

    columns, rows = _grid_shape(len(replays))
    grid_left = 24
    grid_right = WIDTH - 24
    grid_top = HEADER_HEIGHT + 16
    grid_bottom = HEIGHT - FOOTER_HEIGHT - 13
    gap_x = 12
    gap_y = 11
    tile_width = (grid_right - grid_left - gap_x * (columns - 1)) // columns
    tile_height = (grid_bottom - grid_top - gap_y * (rows - 1)) // rows
    route_progress = float(
        np.clip((elapsed_s - 1.1) / max(duration_s - 2.2, 1.0e-6), 0.0, 1.0)
    )
    route_progress = route_progress * route_progress * (3.0 - 2.0 * route_progress)

    for index, replay in enumerate(replays):
        column = index % columns
        row = index // columns
        left = grid_left + column * (tile_width + gap_x)
        top = grid_top + row * (tile_height + gap_y)
        right = left + tile_width
        bottom = top + tile_height
        active = elapsed_s >= 0.12 + index * 0.035
        won = replay.objective_improvement > 1.0e-9
        tied = abs(replay.objective_improvement) <= 1.0e-9
        border = GREEN if won else MUTED if tied else AMBER
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=10,
            fill=PANEL_ACTIVE if active else PANEL,
            outline=border if active else RULE,
            width=2,
        )
        draw.text(
            (left + 12, top + 8),
            f"W{replay.seed:02d}",
            font=bold_font,
            fill=TEXT if active else MUTED,
        )
        draw.text(
            (left + 70, top + 11),
            f"Δ {replay.objective_improvement:+5.1f}",
            font=mono_font,
            fill=border,
        )
        draw.text(
            (right - 91, top + 11),
            f"{replay.ember_bytes / 1000:3.1f}kB",
            font=tiny_font,
            fill=CYAN,
        )
        draw.line(
            (left + 9, top + 35, right - 9, top + 35),
            fill=RULE,
            width=1,
        )
        graph_bounds = (left + 7, top + 38, right - 7, bottom - 10)
        if not active:
            continue

        time_fraction = float(
            np.clip((elapsed_s - 1.2) / max(duration_s - 2.0, 1.0e-6), 0.0, 1.0)
        )
        hazard_index = min(
            round(time_fraction * (replay.hazard.shape[0] - 1)),
            replay.hazard.shape[0] - 1,
        )
        hazard = replay.hazard[hazard_index]
        hazard_low = float(np.percentile(replay.hazard, 10.0))
        hazard_high = float(np.percentile(replay.hazard, 97.0))

        for left_node, right_node in replay.edges:
            draw.line(
                (
                    _point(replay.positions[int(left_node)], graph_bounds),
                    _point(replay.positions[int(right_node)], graph_bounds),
                ),
                fill=(38, 57, 74),
                width=1,
            )
        for node, value in enumerate(hazard):
            x, y = _point(replay.positions[node], graph_bounds)
            color = _hazard_color(float(value), hazard_low, hazard_high)
            radius = 3 if node not in replay.survivors else 5
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=color,
            )
        for node in replay.exits:
            x, y = _point(replay.positions[node], graph_bounds)
            draw.rectangle((x - 5, y - 5, x + 5, y + 5), fill=GREEN)

        for route in replay.flooding_routes:
            points = [_point(replay.positions[node], graph_bounds) for node in route]
            _draw_dashed(draw, points, (132, 68, 79), 2)
        for route in replay.ember_routes:
            points = [_point(replay.positions[node], graph_bounds) for node in route]
            if len(points) > 1:
                draw.line(points, fill=CYAN, width=3, joint="curve")

        for survivor_index, route in enumerate(replay.flooding_routes):
            point = _route_point(replay, route, route_progress, graph_bounds)
            if point is not None:
                x, y = point
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=RED, width=2)
            ember_point = _route_point(
                replay,
                replay.ember_routes[survivor_index],
                route_progress,
                graph_bounds,
            )
            if ember_point is not None:
                x, y = ember_point
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=CYAN)
                draw.ellipse(
                    (x - 8, y - 8, x + 8, y + 8),
                    outline=(64, 136, 154),
                    width=1,
                )

        pulse = (elapsed_s * 1.7 + replay.seed * 0.19) % 1.0
        if pulse < 0.72:
            radius = round(4 + 17 * pulse)
            for robot in replay.robots[::3]:
                x, y = _point(replay.positions[robot], graph_bounds)
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    outline=(43, 116, 137),
                    width=1,
                )

    footer_top = HEIGHT - FOOTER_HEIGHT
    draw.rectangle((0, footer_top, WIDTH, HEIGHT), fill=(9, 15, 26))
    draw.line((0, footer_top, WIDTH, footer_top), fill=RULE, width=2)
    stats = (
        (
            f"{summary.world_count:02d}",
            "WORLDS",
            CYAN,
        ),
        (
            f"{summary.survivor_count:02d}",
            "SURVIVORS",
            TEXT,
        ),
        (
            f"{summary.ember_wins}-{summary.ties}-{summary.flooding_wins}",
            "EMBER · TIE · FLOOD",
            GREEN,
        ),
        (
            f"{summary.mean_ember_objective:4.1f} / "
            f"{summary.mean_flooding_objective:4.1f}",
            "ROUTE COST · E / FLOOD",
            TEXT,
        ),
        (
            f"-{summary.byte_reduction:.1%}",
            "RADIO BYTES",
            CYAN,
        ),
    )
    cell_width = WIDTH // len(stats)
    for index, (value, label, color) in enumerate(stats):
        x = index * cell_width + 28
        draw.text((x, footer_top + 15), value, font=bold_font, fill=color)
        draw.text((x, footer_top + 51), label, font=tiny_font, fill=MUTED)
    draw.text(
        (WIDTH - 345, HEIGHT - 25),
        "CYAN: EMBER + VoI   RED: FLOODING",
        font=tiny_font,
        fill=MUTED,
    )
    return np.asarray(image, dtype=np.uint8)


def render_world_wall(
    replays: tuple[WorldReplay, ...],
    path: Path,
    *,
    poster_path: Path | None = None,
    duration_s: float = 12.0,
    fps: int = 24,
) -> None:
    if duration_s <= 0.0 or fps < 1:
        raise ValueError("duration and frame rate must be positive")
    frame_count = max(1, round(duration_s * fps))
    with VideoWriter(path, fps=fps) as writer:
        for frame in range(frame_count):
            writer.append(
                render_frame(
                    replays,
                    frame / fps,
                    duration_s=duration_s,
                )
            )
    if poster_path is not None:
        from PIL import Image

        poster_path.parent.mkdir(parents=True, exist_ok=True)
        frame = render_frame(replays, duration_s - 0.1, duration_s=duration_s)
        Image.fromarray(frame).save(poster_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-seeds", default="0:19")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/demo/20_world_command_center.mp4"),
    )
    parser.add_argument(
        "--poster",
        type=Path,
        default=Path("outputs/demo/20_world_command_center.png"),
    )
    args = parser.parse_args(argv)
    seeds = parse_seed_spec(args.world_seeds)
    _grid_shape(len(seeds))
    print(f"building {len(seeds)} paired procedural worlds...")
    replays = build_replays(seeds, noise_seed=args.noise_seed)
    summary = _summary(replays)
    print(
        "rendering "
        f"{args.duration:.1f}s at {args.fps} fps "
        f"({summary.survivor_count} survivor routes)..."
    )
    render_world_wall(
        replays,
        args.out,
        poster_path=args.poster,
        duration_s=args.duration,
        fps=args.fps,
    )
    print(
        "world outcomes: "
        f"{summary.ember_wins}-{summary.ties}-{summary.flooding_wins} "
        "(EMBER-tie-flooding)"
    )
    print(f"radio reduction: {summary.byte_reduction:.1%}")
    print(f"video:  {args.out}")
    print(f"poster: {args.poster}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
