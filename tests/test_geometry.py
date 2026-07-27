"""Property tests for the immutable geometry primitives."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cascade.geometry import Axis, Rect, shared_wall

_COORD = st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)


@st.composite
def rects(draw: st.DrawFn, min_side: float = 0.5) -> Rect:
    x0 = draw(_COORD)
    y0 = draw(_COORD)
    w = draw(st.floats(min_value=min_side, max_value=40.0))
    h = draw(st.floats(min_value=min_side, max_value=40.0))
    return Rect(x0, y0, x0 + w, y0 + h)


def test_degenerate_rect_is_rejected() -> None:
    with pytest.raises(ValueError):
        Rect(1.0, 0.0, 0.0, 1.0)


@given(rects())
def test_area_equals_width_times_height(rect: Rect) -> None:
    assert rect.area == pytest.approx(rect.width * rect.height)


@given(rects())
def test_split_conserves_area_and_tiles_parent(rect: Rect) -> None:
    """A split must partition the parent exactly -- no gap, no overlap."""
    for axis in (Axis.X, Axis.Y):
        lo_edge = rect.x0 if axis is Axis.X else rect.y0
        at = lo_edge + 0.5 * rect.extent(axis)
        lo, hi = rect.split(axis, at)
        assert lo.area + hi.area == pytest.approx(rect.area, rel=1e-9)
        assert not lo.overlaps(hi)


@given(rects())
def test_split_outside_bounds_raises(rect: Rect) -> None:
    with pytest.raises(ValueError):
        rect.split(Axis.X, rect.x1 + 1.0)


@given(rects())
def test_aspect_is_at_least_one(rect: Rect) -> None:
    assert rect.aspect >= 1.0 - 1e-9


@given(rects())
def test_inset_never_inverts(rect: Rect) -> None:
    """A large inset clamps to a degenerate centre rather than inverting."""
    shrunk = rect.inset(1000.0)
    assert shrunk.width >= 0.0 and shrunk.height >= 0.0
    assert rect.contains_point(*shrunk.center)


def test_shared_wall_detects_side_by_side_rooms() -> None:
    left = Rect(0.0, 0.0, 4.0, 3.0)
    right = Rect(4.0, 0.0, 8.0, 3.0)
    wall = shared_wall(left, right, min_length=0.9)
    assert wall is not None
    assert wall.axis is Axis.X
    assert wall.at == pytest.approx(4.0)
    assert wall.length == pytest.approx(3.0)


def test_shared_wall_detects_stacked_rooms() -> None:
    lower = Rect(0.0, 0.0, 4.0, 3.0)
    upper = Rect(0.0, 3.0, 4.0, 6.0)
    wall = shared_wall(lower, upper, min_length=0.9)
    assert wall is not None
    assert wall.axis is Axis.Y
    assert wall.at == pytest.approx(3.0)


def test_corner_touching_rooms_share_no_wall() -> None:
    """Rooms meeting at a point must not get a door hung between them."""
    a = Rect(0.0, 0.0, 4.0, 3.0)
    b = Rect(4.0, 3.0, 8.0, 6.0)
    assert shared_wall(a, b, min_length=0.9) is None


def test_shared_wall_respects_min_length() -> None:
    """A shared edge shorter than a door is not a doorway candidate."""
    a = Rect(0.0, 0.0, 4.0, 3.0)
    b = Rect(4.0, 2.5, 8.0, 6.0)  # overlap of only 0.5 m
    assert shared_wall(a, b, min_length=0.9) is None
    assert shared_wall(a, b, min_length=0.4) is not None


def test_separated_rooms_share_no_wall() -> None:
    a = Rect(0.0, 0.0, 4.0, 3.0)
    b = Rect(5.0, 0.0, 9.0, 3.0)
    assert shared_wall(a, b, min_length=0.5) is None


@given(rects(), rects())
def test_shared_wall_is_symmetric(a: Rect, b: Rect) -> None:
    left = shared_wall(a, b, min_length=0.3)
    right = shared_wall(b, a, min_length=0.3)
    assert (left is None) == (right is None)
