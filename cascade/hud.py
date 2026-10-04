"""The frame around the render: captions, telemetry, and video encoding.

Split from `cascade.cinema`, which is about the three-dimensional scene --
what is visible, where the camera is, what is drawn into it.  Everything here
works on the finished pixels instead, and the two have no reason to share a
module beyond having once been written on the same afternoon.

The overlay exists because a render of a rescue swarm is not self-explanatory.
Four coloured dots moving through a grey ruin does not tell you which storey
anything is on, which of them is being followed, how far through its route it
is, or whether it is driving, waiting its turn, or stuck.  All of that is
known to the simulation and none of it is visible, so it is drawn.

It is drawn *around* the panes rather than over them.  Laid on top, the header
and the status block covered the top and bottom third of the plan pane, which
is where the building was: the view whose job is to show every robot at once
was showing the middle half of the floor and hiding the rest behind the
caption describing it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .look import (
    ACCENT,
    INK,
    INK_RAISED,
    RULE,
    STATUS_COLOURS,
    TEXT_DIM,
    TEXT_MUTED,
    TEXT_PRIMARY,
)

#: Label and numeral faces.  Two families, used for two jobs: proportional for
#: prose, monospace for anything that changes every frame -- a clock set in a
#: proportional face jitters horizontally as its digits change, which draws the
#: eye to the least interesting thing in the picture.
_LABEL_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)
_BODY_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
)
_MONO_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
)

# H.264 is most portable when both dimensions are divisible by a macroblock.
# imageio otherwise inserts an implicit ffmpeg scale filter, changing the
# picture without the caller asking it to.  VideoWriter pads instead so no
# scene or HUD pixels are discarded or interpolated.
VIDEO_MACROBLOCK: int = 2


def _font(candidates: tuple[str, ...], size: int):
    from PIL import ImageFont

    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


@dataclass(frozen=True, slots=True)
class RobotReadout:
    """One line of telemetry."""

    name: str
    livery: tuple[float, float, float]
    storey: int
    goal_storey: int
    progress: float
    status: str
    goal_label: str = ""
    behaviour_label: str | None = None
    #: True for the robot the hero camera is currently following.
    is_subject: bool = False

    @property
    def rgb(self) -> tuple[int, int, int]:
        r, g, b = self.livery
        return (int(255 * r), int(255 * g), int(255 * b))

    @property
    def behaviour(self) -> str:
        """Plain-language state for viewers who do not know controller terms."""

        return self.behaviour_label or {
            "waiting": "DEPLOYING",
            "driving": "FOLLOWING ROUTE",
            "climbing": "CLIMBING",
            "recovering": "BACKING OUT",
            "fallen": "IMMOBILIZED",
            "stalled": "BLOCKED",
            "unroutable": "NO SAFE ROUTE",
            "arrived": "TARGET REACHED",
        }.get(self.status, self.status.upper())


@dataclass(frozen=True, slots=True)
class ResearchMetric:
    """One large, auditable value in the live research strip."""

    label: str
    value: str
    detail: str = ""
    colour: tuple[int, int, int] = ACCENT


@dataclass(frozen=True, slots=True)
class _Metrics:
    """Every size in the overlay, derived from one number.

    Deriving them rather than hard-coding them is what lets `--width 1920` and
    `--width 800` both produce a frame with the same proportions instead of one
    with unreadable text and one with a caption bar taller than the picture.
    """

    width: int
    n_rows: int

    @property
    def pad(self) -> int:
        return max(14, self.width // 90)

    @property
    def title_size(self) -> int:
        return max(18, self.width // 44)

    @property
    def body_size(self) -> int:
        return max(12, self.width // 78)

    @property
    def small_size(self) -> int:
        return max(11, self.width // 96)

    @property
    def row_height(self) -> int:
        return self.body_size + max(10, self.width // 130)

    @property
    def header_height(self) -> int:
        return self.pad * 2 + self.title_size + self.small_size + 4

    @property
    def panel_height(self) -> int:
        return self.pad + self.small_size + 6 + self.row_height * self.n_rows + self.pad


def draw_hud(
    frame: np.ndarray,
    *,
    title: str,
    subtitle: str,
    readouts: list[RobotReadout],
    elapsed_s: float,
    n_storeys: int = 0,
    mission_status: str | None = None,
    show_storeys: bool = True,
    research_metrics: list[ResearchMetric] | None = None,
) -> np.ndarray:
    """Return `frame` with a header band above it and a telemetry block below."""
    from PIL import Image, ImageDraw

    pane = Image.fromarray(frame).convert("RGB")
    width, pane_height = pane.size
    metrics = _Metrics(
        width=width,
        n_rows=1 if research_metrics is not None else len(readouts),
    )

    image = Image.new(
        "RGB",
        (width, metrics.header_height + pane_height + metrics.panel_height),
        INK,
    )
    image.paste(pane, (0, metrics.header_height))
    draw = ImageDraw.Draw(image, "RGBA")

    arrived = sum(1 for r in readouts if r.status == "arrived")
    _draw_header(
        draw,
        metrics,
        title=title,
        subtitle=subtitle,
        elapsed_s=elapsed_s,
        arrived=arrived,
        total=len(readouts),
        mission_status=mission_status,
    )
    if research_metrics is not None:
        _draw_metrics_panel(
            draw,
            metrics,
            top=metrics.header_height + pane_height,
            metrics=research_metrics,
        )
    else:
        _draw_panel(
            draw,
            metrics,
            top=metrics.header_height + pane_height,
            readouts=readouts,
            n_storeys=n_storeys or _storey_span(readouts),
            show_storeys=show_storeys,
        )
    return np.asarray(image, dtype=np.uint8)


def _storey_span(readouts: list[RobotReadout]) -> int:
    highest = max((max(r.storey, r.goal_storey) for r in readouts), default=0)
    return highest + 1


def _draw_header(
    draw,
    m: _Metrics,
    *,
    title,
    subtitle,
    elapsed_s,
    arrived,
    total,
    mission_status,
):
    """Title and mission state, over a hairline rule."""
    title_font = _font(_LABEL_FONTS, m.title_size)
    small_font = _font(_BODY_FONTS, m.small_size)
    mono_font = _font(_MONO_FONTS, m.body_size)
    tick = max(3, m.width // 400)

    # The accent bar is the only saturated thing in the band, so it anchors the
    # top-left corner and the title hangs off it.
    draw.rectangle(
        [m.pad, m.pad, m.pad + tick, m.pad + m.title_size + m.small_size],
        fill=(*ACCENT, 255),
    )
    text_x = m.pad + tick + m.pad // 2
    draw.text((text_x, m.pad - 2), title, font=title_font, fill=TEXT_PRIMARY)
    draw.text(
        (text_x, m.pad + m.title_size + 2), subtitle, font=small_font, fill=TEXT_MUTED
    )

    clock = f"T+{elapsed_s:6.1f} s"
    tally = mission_status or f"{arrived}/{total} ARRIVED"
    right = m.width - m.pad
    draw.text(
        (right - draw.textlength(clock, font=mono_font), m.pad - 2),
        clock,
        font=mono_font,
        fill=TEXT_PRIMARY,
    )
    draw.text(
        (right - draw.textlength(tally, font=small_font), m.pad + m.title_size + 2),
        tally,
        font=small_font,
        fill=(
            STATUS_COLOURS["fallen"]
            if mission_status and mission_status.startswith("RUN FAILED")
            else ACCENT
            if arrived or mission_status
            else TEXT_DIM
        ),
    )
    draw.line(
        [0, m.header_height - 1, m.width, m.header_height - 1], fill=RULE, width=1
    )


#: Column positions in the telemetry block, as fractions of the frame width.
#: Fixed so the columns line up down the block whatever the names happen to be.
_COL_NAME: float = 0.0
_COL_LADDER: float = 0.115
_COL_BAR: float = 0.30
_COL_DESTINATION: float = 0.56
_COL_STATUS: float = 0.80


def _draw_panel(draw, m: _Metrics, *, top, readouts, n_storeys, show_storeys) -> None:
    """One row per robot: livery, name, storey ladder, progress, status."""
    label_font = _font(_LABEL_FONTS, m.body_size)
    small_font = _font(_BODY_FONTS, m.small_size)
    mono_font = _font(_MONO_FONTS, m.small_size)

    draw.rectangle([0, top, m.width, top + m.panel_height], fill=INK_RAISED)
    inner = m.width - 2 * m.pad

    def column(fraction: float) -> int:
        return m.pad + int(inner * fraction)

    head_y = top + m.pad // 2
    columns = (
        (
            ("ROBOT", _COL_NAME),
            ("STOREY", _COL_LADDER),
            ("ROUTE FLOWN", _COL_BAR),
            ("DESTINATION", _COL_DESTINATION),
            ("STATE", _COL_STATUS),
        )
        if show_storeys
        else (
            ("AGENT", _COL_NAME),
            ("CURRENT ROUTE", 0.16),
            ("ASSIGNED SURVIVOR", 0.50),
            ("BEHAVIOUR", _COL_STATUS),
        )
    )
    for label, fraction in columns:
        draw.text((column(fraction), head_y), label, font=small_font, fill=TEXT_DIM)

    for index, readout in enumerate(readouts):
        y = head_y + m.small_size + 6 + index * m.row_height
        _draw_row(
            draw,
            m,
            y=y,
            column=column,
            readout=readout,
            n_storeys=n_storeys,
            fonts=(label_font, small_font, mono_font),
            show_storeys=show_storeys,
        )


def _draw_metrics_panel(draw, m: _Metrics, *, top, metrics) -> None:
    """Replace low-value progress rows with a few legible research measures."""

    label_font = _font(_LABEL_FONTS, m.small_size)
    value_font = _font(_MONO_FONTS, m.body_size)
    detail_font = _font(_BODY_FONTS, m.small_size)
    draw.rectangle([0, top, m.width, top + m.panel_height], fill=INK_RAISED)
    count = max(1, len(metrics))
    inner = m.width - 2 * m.pad
    cell = inner / count
    for index, metric in enumerate(metrics):
        left = int(m.pad + index * cell)
        if index:
            draw.line(
                [
                    left - m.pad // 2,
                    top + m.pad,
                    left - m.pad // 2,
                    top + m.panel_height - m.pad,
                ],
                fill=RULE,
                width=1,
            )
        draw.text(
            (left, top + m.pad // 2),
            metric.label.upper(),
            font=label_font,
            fill=TEXT_DIM,
        )
        value_y = top + m.pad // 2 + m.small_size + 5
        draw.text(
            (left, value_y),
            metric.value,
            font=value_font,
            fill=(*metric.colour, 255),
        )
        if metric.detail:
            draw.text(
                (left, value_y + m.body_size + 2),
                metric.detail,
                font=detail_font,
                fill=TEXT_MUTED,
            )


def _draw_row(
    draw,
    m: _Metrics,
    *,
    y,
    column,
    readout,
    n_storeys,
    fonts,
    show_storeys=True,
) -> None:
    label_font, small_font, mono_font = fonts
    colour = readout.rgb
    mid = y + m.body_size // 2

    if readout.is_subject:
        # The hero pane follows one robot at a time and never says which.
        draw.rectangle(
            [m.pad // 3, y - 3, m.pad // 3 + 3, y + m.body_size + 3],
            fill=(*ACCENT, 255),
        )
    draw.rectangle(
        [column(_COL_NAME), y + 2, column(_COL_NAME) + 11, y + m.body_size],
        fill=(*colour, 255),
    )
    draw.text(
        (column(_COL_NAME) + 19, y - 1),
        readout.name,
        font=label_font,
        fill=TEXT_PRIMARY if readout.is_subject else TEXT_MUTED,
    )

    if show_storeys:
        _draw_ladder(
            draw,
            x=column(_COL_LADDER),
            mid=mid,
            step=max(9, m.width // 105),
            n_storeys=n_storeys,
            readout=readout,
        )

    # The bar has to give back whatever the percentage needs, measured rather
    # than guessed at: a fixed gap that fits "4%" puts "100%" on top of the
    # status word in the next column.
    percent = f"{100 * readout.progress:3.0f}%"
    percent_w = draw.textlength(percent, font=mono_font)
    bar_x = column(_COL_BAR if show_storeys else 0.16)
    destination_column = _COL_DESTINATION if show_storeys else 0.50
    bar_w = int(column(destination_column) - bar_x - percent_w - 2 * m.pad)
    bar_h = max(6, m.body_size // 2)
    draw.rectangle(
        [bar_x, mid - bar_h // 2, bar_x + bar_w, mid + bar_h // 2], fill=RULE
    )
    filled = int(bar_w * max(0.0, min(1.0, readout.progress)))
    if filled:
        draw.rectangle(
            [bar_x, mid - bar_h // 2, bar_x + filled, mid + bar_h // 2],
            fill=(*colour, 255),
        )
    draw.text((bar_x + bar_w + m.pad, y), percent, font=mono_font, fill=TEXT_MUTED)

    draw.text(
        (column(destination_column), y - 1),
        readout.goal_label[:22],
        font=small_font,
        fill=(*readout.rgb, 255),
    )

    draw.text(
        (column(_COL_STATUS), y - 1),
        readout.status if show_storeys else readout.behaviour,
        font=small_font,
        fill=STATUS_COLOURS.get(readout.status, TEXT_MUTED),
    )


def _draw_ladder(draw, *, x, mid, step, n_storeys, readout) -> None:
    """A rung per storey: filled where the robot is, ringed where it is bound.

    A ladder rather than the text it replaces.  "storey 2 -> 3" is four robots'
    worth of small print that has to be read one line at a time; four ladders
    stacked in a column show the swarm spread over the building at a glance,
    which is the single fact the demo exists to establish.
    """
    size = step - 3
    for storey in range(max(1, n_storeys)):
        left = x + storey * step
        box = [left, mid - size // 2, left + size, mid + size // 2]
        if storey == readout.storey:
            draw.rectangle(box, fill=(*readout.rgb, 255))
        elif storey == readout.goal_storey:
            draw.rectangle(box, outline=(*ACCENT, 255), width=1)
        else:
            draw.rectangle(box, fill=RULE)


def research_card(
    frame: np.ndarray,
    *,
    eyebrow: str,
    title: str,
    body: str,
    metrics: list[ResearchMetric] | None = None,
    footer: str = "",
) -> np.ndarray:
    """Replace a frame with a paced hypothesis or results card."""
    from PIL import Image, ImageDraw

    height, width = frame.shape[:2]
    image = Image.new("RGB", (width, height), INK)
    draw = ImageDraw.Draw(image, "RGBA")
    pad = max(28, width // 24)
    eyebrow_font = _font(_LABEL_FONTS, max(15, width // 96))
    title_font = _font(_LABEL_FONTS, max(30, width // 34))
    body_font = _font(_BODY_FONTS, max(18, width // 58))
    metric_label_font = _font(_LABEL_FONTS, max(14, width // 110))
    metric_value_font = _font(_MONO_FONTS, max(24, width // 46))
    detail_font = _font(_BODY_FONTS, max(13, width // 105))

    draw.rectangle(
        [pad, pad, pad + max(5, width // 320), height - pad], fill=(*ACCENT, 255)
    )
    left = pad + width // 36
    draw.text((left, pad), eyebrow.upper(), font=eyebrow_font, fill=ACCENT)
    title_y = pad + width // 40
    draw.text((left, title_y), title, font=title_font, fill=TEXT_PRIMARY)
    body_y = title_y + width // 24
    draw.multiline_text(
        (left, body_y), body, font=body_font, fill=TEXT_MUTED, spacing=8
    )

    cards = metrics or []
    if cards:
        cards_y = int(height * 0.58)
        available = width - left - pad
        cell = available / len(cards)
        for index, metric in enumerate(cards):
            x = int(left + index * cell)
            if index:
                draw.line(
                    [x - pad // 4, cards_y, x - pad // 4, cards_y + height // 7],
                    fill=RULE,
                    width=1,
                )
            draw.text(
                (x, cards_y),
                metric.label.upper(),
                font=metric_label_font,
                fill=TEXT_DIM,
            )
            draw.text(
                (x, cards_y + width // 80),
                metric.value,
                font=metric_value_font,
                fill=(*metric.colour, 255),
            )
            if metric.detail:
                draw.text(
                    (x, cards_y + width // 80 + width // 42),
                    metric.detail,
                    font=detail_font,
                    fill=TEXT_MUTED,
                )
    if footer:
        draw.text(
            (left, height - pad - width // 90), footer, font=detail_font, fill=TEXT_DIM
        )
    return np.asarray(image, dtype=np.uint8)


def caption(frame: np.ndarray, text: str, *, accent: bool = False) -> np.ndarray:
    """Lay a small label chip in the bottom-left corner of one pane.

    Each pane is showing something different and non-obvious -- a cut, a slice
    -- and naming them in the header meant a run-on sentence describing two
    pictures neither of which it was next to.
    """
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image, "RGBA")

    size = max(11, width // 62)
    font = _font(_LABEL_FONTS, size)
    pad = max(6, width // 90)
    text_w = draw.textlength(text, font=font)
    box = [pad, height - pad - size - pad, pad + text_w + 2 * pad, height - pad]

    draw.rectangle(box, fill=(*INK, 200))
    draw.line([box[0], box[1], box[0], box[3]], fill=(*ACCENT, 255), width=2)
    draw.text((box[0] + pad, box[1] + pad // 2 - 1), text, font=font, fill=TEXT_PRIMARY)
    return np.asarray(image, dtype=np.uint8)


def stack_panes(
    hero: np.ndarray,
    plan: np.ndarray,
    *,
    plan_fraction: float | None = None,
    target_width: int | None = None,
    hero_label: str | None = None,
    plan_label: str | None = None,
) -> np.ndarray:
    """Put the plan beside the hero, optionally making it the primary pane.

    Labels are applied after the cover-crop. Applying them to source panes
    first lets the crop erase the words identifying an evaluation view.
    """
    from PIL import Image

    target_h = hero.shape[0]
    gap_w = 4
    if plan_fraction is None:
        scale = target_h / plan.shape[0]
        plan_w = max(1, int(plan.shape[1] * scale))
        hero_w = hero.shape[1]
    else:
        if not 0.0 < plan_fraction < 1.0:
            raise ValueError("plan_fraction must be between zero and one")
        total = target_width or hero.shape[1] + plan.shape[1]
        plan_w = max(1, round((total - gap_w) * plan_fraction))
        hero_w = max(1, total - gap_w - plan_w)

    def cover(frame: np.ndarray, width: int) -> np.ndarray:
        source = Image.fromarray(frame)
        scale = max(width / source.width, target_h / source.height)
        resized = source.resize(
            (
                max(width, round(source.width * scale)),
                max(target_h, round(source.height * scale)),
            ),
            Image.LANCZOS,
        )
        left = (resized.width - width) // 2
        top = (resized.height - target_h) // 2
        return np.asarray(
            resized.crop((left, top, left + width, top + target_h)),
            dtype=np.uint8,
        )

    hero_image = cover(hero, hero_w)
    plan_image = cover(plan, plan_w)
    if hero_label:
        hero_image = caption(hero_image, hero_label)
    if plan_label:
        plan_image = caption(plan_image, plan_label)
    gap = np.full((target_h, gap_w, 3), RULE, dtype=np.uint8)
    return np.concatenate((hero_image, gap, plan_image), axis=1)


class VideoWriter:
    """Encodes frames to an mp4 as they arrive.

    Streaming rather than collecting, because a traversal of a four-storey
    building runs to a couple of thousand frames and a pane pair at this size
    is 4.7 MB each.  Holding them all and encoding at the end needs the better
    part of ten gigabytes, and the demo was killed by the OOM reaper after
    twenty minutes of simulation with nothing written out.
    """

    def __init__(self, path: str | Path, fps: int = 30) -> None:
        import imageio.v2 as iio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = iio.get_writer(
            self.path,
            fps=fps,
            codec="libx264",
            quality=8,
            # Frames are padded explicitly below.  Disabling imageio's own
            # macroblock handling guarantees it cannot silently add a scale
            # filter if that invariant is accidentally broken later.
            macro_block_size=1,
        )
        self._source_shape: tuple[int, int, int] | None = None
        self.frames = 0

    def append(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("video frames must be H x W x 3 uint8 RGB arrays")
        if not frame.shape[0] or not frame.shape[1]:
            raise ValueError("video frames must have non-zero dimensions")
        if self._source_shape is None:
            self._source_shape = frame.shape
        elif frame.shape != self._source_shape:
            raise ValueError(
                f"video frame shape changed from {self._source_shape} to {frame.shape}"
            )

        self._writer.append_data(_encoder_safe_frame(frame))
        self.frames += 1

    def close(self) -> Path | None:
        self._writer.close()
        return self.path if self.frames else None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _encoder_safe_frame(frame: np.ndarray) -> np.ndarray:
    """Pad an RGB frame at its right and bottom edges to a full macroblock.

    Padding rather than scaling retains every source pixel and therefore the
    camera composition.  The at-most-15-pixel border uses the HUD background,
    so it reads as part of the existing frame rather than video garbage.
    """
    height, width = frame.shape[:2]
    encoded_height = ((height + VIDEO_MACROBLOCK - 1) // VIDEO_MACROBLOCK) * (
        VIDEO_MACROBLOCK
    )
    encoded_width = ((width + VIDEO_MACROBLOCK - 1) // VIDEO_MACROBLOCK) * (
        VIDEO_MACROBLOCK
    )
    if (encoded_height, encoded_width) == (height, width):
        return np.ascontiguousarray(frame)

    padded = np.full((encoded_height, encoded_width, 3), INK, dtype=np.uint8)
    padded[:height, :width] = frame
    return padded
