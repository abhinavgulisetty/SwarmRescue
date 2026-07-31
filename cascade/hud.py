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
    #: True for the robot the hero camera is currently following.
    is_subject: bool = False

    @property
    def rgb(self) -> tuple[int, int, int]:
        r, g, b = self.livery
        return (int(255 * r), int(255 * g), int(255 * b))


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
) -> np.ndarray:
    """Return `frame` with a header band above it and a telemetry block below."""
    from PIL import Image, ImageDraw

    pane = Image.fromarray(frame).convert("RGB")
    width, pane_height = pane.size
    metrics = _Metrics(width=width, n_rows=len(readouts))

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
    )
    _draw_panel(
        draw,
        metrics,
        top=metrics.header_height + pane_height,
        readouts=readouts,
        n_storeys=n_storeys or _storey_span(readouts),
    )
    return np.asarray(image, dtype=np.uint8)


def _storey_span(readouts: list[RobotReadout]) -> int:
    highest = max((max(r.storey, r.goal_storey) for r in readouts), default=0)
    return highest + 1


def _draw_header(draw, m: _Metrics, *, title, subtitle, elapsed_s, arrived, total):
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
    tally = f"{arrived}/{total} ARRIVED"
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
        fill=ACCENT if arrived else TEXT_DIM,
    )
    draw.line(
        [0, m.header_height - 1, m.width, m.header_height - 1], fill=RULE, width=1
    )


#: Column positions in the telemetry block, as fractions of the frame width.
#: Fixed so the columns line up down the block whatever the names happen to be.
_COL_NAME: float = 0.0
_COL_LADDER: float = 0.115
_COL_BAR: float = 0.30
_COL_STATUS: float = 0.56


def _draw_panel(draw, m: _Metrics, *, top, readouts, n_storeys) -> None:
    """One row per robot: livery, name, storey ladder, progress, status."""
    label_font = _font(_LABEL_FONTS, m.body_size)
    small_font = _font(_BODY_FONTS, m.small_size)
    mono_font = _font(_MONO_FONTS, m.small_size)

    draw.rectangle([0, top, m.width, top + m.panel_height], fill=INK_RAISED)
    inner = m.width - 2 * m.pad

    def column(fraction: float) -> int:
        return m.pad + int(inner * fraction)

    head_y = top + m.pad // 2
    for label, fraction in (
        ("ROBOT", _COL_NAME),
        ("STOREY", _COL_LADDER),
        ("ROUTE FLOWN", _COL_BAR),
        ("STATE", _COL_STATUS),
    ):
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
        )


def _draw_row(draw, m: _Metrics, *, y, column, readout, n_storeys, fonts) -> None:
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
    bar_x = column(_COL_BAR)
    bar_w = int(column(_COL_STATUS) - bar_x - percent_w - 2 * m.pad)
    bar_h = max(6, m.body_size // 2)
    draw.rectangle([bar_x, mid - bar_h // 2, bar_x + bar_w, mid + bar_h // 2], fill=RULE)
    filled = int(bar_w * max(0.0, min(1.0, readout.progress)))
    if filled:
        draw.rectangle(
            [bar_x, mid - bar_h // 2, bar_x + filled, mid + bar_h // 2],
            fill=(*colour, 255),
        )
    draw.text((bar_x + bar_w + m.pad, y), percent, font=mono_font, fill=TEXT_MUTED)

    draw.text(
        (column(_COL_STATUS), y - 1),
        readout.status,
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


def stack_panes(hero: np.ndarray, plan: np.ndarray) -> np.ndarray:
    """Put the plan pane beside the hero shot, matching heights."""
    from PIL import Image

    target_h = hero.shape[0]
    scale = target_h / plan.shape[0]
    resized = np.asarray(
        Image.fromarray(plan).resize(
            (max(1, int(plan.shape[1] * scale)), target_h), Image.LANCZOS
        ),
        dtype=np.uint8,
    )
    gap = np.full((target_h, 4, 3), RULE, dtype=np.uint8)
    return np.concatenate((hero, gap, resized), axis=1)


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
        self._writer = iio.get_writer(self.path, fps=fps, codec="libx264", quality=8)
        self._shape: tuple[int, int] | None = None
        self.frames = 0

    def append(self, frame: np.ndarray) -> None:
        if self._shape is None:
            height, width = frame.shape[:2]
            # H.264 wants even dimensions, and the encoder needs every frame
            # the same size as the first.
            self._shape = (height - height % 2, width - width % 2)
        height, width = self._shape
        self._writer.append_data(np.ascontiguousarray(frame[:height, :width]))
        self.frames += 1

    def close(self) -> Path | None:
        self._writer.close()
        return self.path if self.frames else None

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
