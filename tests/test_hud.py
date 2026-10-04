"""Tests for the overlay drawn on the finished frame.

These look like cosmetic tests and are not.  The first cut of the HUD was laid
*over* the render, and between them the title bar and the status block covered
the top and bottom third of the plan pane -- which is where the building was.
The view whose entire job is to show where every robot is was showing the
middle half of the floor and hiding the rest behind the caption describing it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cascade.hud import (
    ResearchMetric,
    RobotReadout,
    VideoWriter,
    caption,
    draw_hud,
    stack_panes,
)
from cascade.look import INK


def _readout(**kwargs) -> RobotReadout:
    fields = {
        "name": "robot0",
        "livery": (1.0, 0.5, 0.1),
        "storey": 0,
        "goal_storey": 3,
        "progress": 0.4,
        "status": "driving",
    }
    return RobotReadout(**{**fields, **kwargs})


def test_hud_adds_bands_without_covering_the_frame() -> None:
    frame = np.full((240, 480, 3), 200, dtype=np.uint8)
    out = draw_hud(
        frame,
        title="t",
        subtitle="s",
        readouts=[_readout()],
        elapsed_s=1.0,
    )
    assert out.dtype == np.uint8
    assert out.shape[1] == frame.shape[1]
    assert out.shape[0] > frame.shape[0]
    kept = sum(1 for row in out.reshape(out.shape[0], -1) if row.min() == 200)
    assert kept == frame.shape[0], "the overlay painted over the render"


def test_hud_actually_draws_something() -> None:
    frame = np.zeros((240, 480, 3), dtype=np.uint8)
    out = draw_hud(
        frame,
        title="EMBER",
        subtitle="s",
        readouts=[_readout()],
        elapsed_s=1.0,
    )
    assert out.any(), "overlay left the frame untouched"


def test_hud_marks_which_robot_the_camera_is_following() -> None:
    """The hero pane follows one robot at a time and never says which."""
    frame = np.zeros((240, 480, 3), dtype=np.uint8)
    plain = draw_hud(
        frame,
        title="EMBER",
        subtitle="s",
        readouts=[_readout(is_subject=False)],
        elapsed_s=1.0,
        n_storeys=5,
    )
    marked = draw_hud(
        frame,
        title="EMBER",
        subtitle="s",
        readouts=[_readout(is_subject=True)],
        elapsed_s=1.0,
        n_storeys=5,
    )
    assert not np.array_equal(plain, marked)


def test_hud_exposes_each_robot_destination() -> None:
    frame = np.zeros((240, 800, 3), dtype=np.uint8)
    unlabeled = draw_hud(
        frame,
        title="EMBER",
        subtitle="s",
        readouts=[_readout()],
        elapsed_s=1.0,
    )
    labeled = draw_hud(
        frame,
        title="EMBER",
        subtitle="s",
        readouts=[_readout(goal_label="S3 · OFFICE-2")],
        elapsed_s=1.0,
    )
    assert not np.array_equal(unlabeled, labeled)


def test_hud_marks_a_terminal_failure_explicitly() -> None:
    frame = np.zeros((240, 800, 3), dtype=np.uint8)
    live = draw_hud(
        frame,
        title="EMBER",
        subtitle="live",
        readouts=[_readout()],
        elapsed_s=1.0,
    )
    failed = draw_hud(
        frame,
        title="EMBER",
        subtitle="live",
        readouts=[_readout(status="fallen")],
        elapsed_s=1.0,
        mission_status="RUN FAILED · ROBOT0 FALLEN · 0/1 ARRIVED",
    )
    assert not np.array_equal(live, failed)


def test_hud_geometry_scales_with_the_frame() -> None:
    """A fixed-pixel overlay is either illegible or enormous, never both."""
    heights = []
    for width in (480, 1600):
        frame = np.zeros((240, width, 3), dtype=np.uint8)
        out = draw_hud(
            frame, title="t", subtitle="s", readouts=[_readout()], elapsed_s=1.0
        )
        heights.append(out.shape[0] - 240)
    assert heights[1] > heights[0]


def test_hud_survives_a_robot_that_never_got_a_route() -> None:
    """An unroutable robot reports storey 0 and goal 0; the ladder needs a rung."""
    frame = np.zeros((120, 480, 3), dtype=np.uint8)
    out = draw_hud(
        frame,
        title="t",
        subtitle="s",
        readouts=[_readout(storey=0, goal_storey=0, progress=1.0, status="arrived")],
        elapsed_s=0.0,
    )
    assert out.shape[0] > 120


def test_caption_writes_into_the_pane_it_labels() -> None:
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    out = caption(frame, "plan of storey 2")
    assert out.shape == frame.shape
    # The chip sits at the bottom; the top of the pane is untouched.
    assert not out[:100].any()
    assert out[100:].any()


def test_stack_panes_joins_side_by_side() -> None:
    left = np.zeros((100, 160, 3), dtype=np.uint8)
    right = np.zeros((50, 50, 3), dtype=np.uint8)
    out = stack_panes(left, right)
    assert out.shape[0] == 100
    assert out.shape[1] > left.shape[1]


def test_stack_panes_applies_labels_after_the_cover_crop() -> None:
    hero = np.zeros((100, 160, 3), dtype=np.uint8)
    plan = np.zeros_like(hero)

    out = stack_panes(
        hero,
        plan,
        plan_fraction=0.65,
        target_width=400,
        hero_label="EVIDENCE CAMERA",
        plan_label="EVALUATION VIEW",
    )

    hero_width = round((400 - 4) * 0.35)
    assert out[-40:, :hero_width].any()
    assert out[-40:, hero_width + 4 :].any()


def test_progress_label_does_not_run_into_the_status_column() -> None:
    """A gap sized for "4%" put "100%" on top of the word next to it."""
    from PIL import Image, ImageDraw

    from cascade.hud import _COL_STATUS, _MONO_FONTS, _draw_row, _font, _Metrics

    metrics = _Metrics(width=1200, n_rows=1)
    inner = metrics.width - 2 * metrics.pad
    status_x = metrics.pad + int(inner * _COL_STATUS)
    mono = _font(_MONO_FONTS, metrics.small_size)

    image = Image.new("RGB", (metrics.width, 60), (0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    _draw_row(
        draw,
        metrics,
        y=10,
        column=lambda f: metrics.pad + int(inner * f),
        readout=_readout(progress=1.0),
        n_storeys=5,
        fonts=(mono, mono, mono),
    )
    # Nothing drawn by the progress column may reach the status column.
    strip = np.asarray(image)[:, status_x - 4 : status_x]
    assert not strip.any(), "the progress label overran into the status column"


def test_every_robot_state_has_an_explicit_hud_colour() -> None:
    from cascade.look import STATUS_COLOURS
    from cascade.robot import RobotState

    assert {state.value for state in RobotState} <= STATUS_COLOURS.keys()


def test_compact_research_layout_is_exactly_full_hd() -> None:
    hero = np.zeros((865, 960, 3), dtype=np.uint8)
    plan = np.zeros_like(hero)
    panes = stack_panes(hero, plan, plan_fraction=0.65, target_width=1920)
    out = draw_hud(
        panes,
        title="EMBER",
        subtitle="evidence",
        readouts=[_readout() for _ in range(4)],
        elapsed_s=1.0,
        mission_status="ILLUSTRATIVE SEED 7",
        research_metrics=[ResearchMetric("metric", "value") for _ in range(4)],
    )
    assert out.shape == (1080, 1920, 3)


class _CaptureWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.closed = False

    def append_data(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def close(self) -> None:
        self.closed = True


def test_video_writer_pads_to_even_dimensions_without_losing_pixels(
    monkeypatch, tmp_path
) -> None:
    import imageio.v2 as iio

    capture = _CaptureWriter()
    options = {}

    def get_writer(path, **kwargs):
        options.update(kwargs)
        return capture

    monkeypatch.setattr(iio, "get_writer", get_writer)
    frame = np.arange(17 * 19 * 3, dtype=np.uint8).reshape(17, 19, 3)

    writer = VideoWriter(tmp_path / "demo.mp4", fps=24)
    writer.append(frame)
    writer.close()

    assert options["macro_block_size"] == 1
    assert capture.closed
    assert capture.frames[0].shape == (18, 20, 3)
    np.testing.assert_array_equal(capture.frames[0][:17, :19], frame)
    assert np.all(capture.frames[0][17:, :] == INK)
    assert np.all(capture.frames[0][:17, 19:] == INK)


def test_video_writer_rejects_a_shape_change_instead_of_cropping(
    monkeypatch, tmp_path
) -> None:
    import imageio.v2 as iio

    monkeypatch.setattr(iio, "get_writer", lambda *args, **kwargs: _CaptureWriter())
    writer = VideoWriter(tmp_path / "demo.mp4")
    writer.append(np.zeros((16, 16, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="shape changed"):
        writer.append(np.zeros((16, 15, 3), dtype=np.uint8))
    writer.close()
