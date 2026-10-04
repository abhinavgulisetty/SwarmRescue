"""Immutable 2-D geometry primitives for the floor-plan grammar.

Everything here is a pure function or a frozen value object: operations return
new `Rect`s rather than editing one in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

#: Tolerance for treating two coordinates as coincident, in metres.  Rect edges
#: are produced by repeated subdivision, so exact float equality is unsafe.
EPS_M: float = 1e-9


class Axis(StrEnum):
    """Split/adjacency axis.  `X` means the shared edge is vertical."""

    X = "x"
    Y = "y"

    @property
    def other(self) -> Axis:
        return Axis.Y if self is Axis.X else Axis.X


@dataclass(frozen=True, slots=True)
class Rect:
    """An axis-aligned rectangle in metres, with `x0 <= x1` and `y0 <= y1`."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"degenerate Rect: {self!r}")

    # -- derived quantities --

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))

    @property
    def aspect(self) -> float:
        """Long side over short side; `inf` for a zero-thickness rectangle."""
        lo, hi = sorted((self.width, self.height))
        return float("inf") if lo <= EPS_M else hi / lo

    def extent(self, axis: Axis) -> float:
        return self.width if axis is Axis.X else self.height

    # -- construction --

    def split(self, axis: Axis, at: float) -> tuple[Rect, Rect]:
        """Split into two rectangles at absolute coordinate `at`."""
        if axis is Axis.X:
            if not (self.x0 < at < self.x1):
                raise ValueError(f"split x={at} outside {self!r}")
            return (
                Rect(self.x0, self.y0, at, self.y1),
                Rect(at, self.y0, self.x1, self.y1),
            )
        if not (self.y0 < at < self.y1):
            raise ValueError(f"split y={at} outside {self!r}")
        return (
            Rect(self.x0, self.y0, self.x1, at),
            Rect(self.x0, at, self.x1, self.y1),
        )

    def inset(self, margin: float) -> Rect:
        """Shrink by `margin` on every side, clamped to a degenerate centre."""
        cx, cy = self.center
        half_w = max(0.0, 0.5 * self.width - margin)
        half_h = max(0.0, 0.5 * self.height - margin)
        return Rect(cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    # -- predicates --

    def contains_point(self, x: float, y: float, tol: float = EPS_M) -> bool:
        return (
            self.x0 - tol <= x <= self.x1 + tol and self.y0 - tol <= y <= self.y1 + tol
        )

    def overlaps(self, other: Rect, tol: float = EPS_M) -> bool:
        """True if the interiors overlap by more than `tol` in both axes."""
        return (
            min(self.x1, other.x1) - max(self.x0, other.x0) > tol
            and min(self.y1, other.y1) - max(self.y0, other.y0) > tol
        )


@dataclass(frozen=True, slots=True)
class SharedWall:
    """The contact segment between two adjacent rooms.

    `axis` is the axis the wall's *normal* runs along: `Axis.X` means the two
    rooms sit side by side and the wall itself is a vertical line at `at`.
    """

    axis: Axis
    at: float
    lo: float
    hi: float

    @property
    def length(self) -> float:
        return self.hi - self.lo

    def midpoint(self) -> tuple[float, float]:
        mid = 0.5 * (self.lo + self.hi)
        return (self.at, mid) if self.axis is Axis.X else (mid, self.at)

    def point_at(self, along: float) -> tuple[float, float]:
        """Point on the wall at absolute coordinate `along` on its long axis."""
        return (self.at, along) if self.axis is Axis.X else (along, self.at)


def shared_wall(a: Rect, b: Rect, min_length: float) -> SharedWall | None:
    """Return the wall shared by `a` and `b`, or None if they are not adjacent.

    Two rectangles are adjacent when they abut on one axis and their extents
    overlap by at least `min_length` on the other.  Rectangles that merely
    touch at a corner share no wall.
    """
    for axis in (Axis.X, Axis.Y):
        lo_a, hi_a = _span(a, axis)
        lo_b, hi_b = _span(b, axis)

        # Abutting means one's high edge coincides with the other's low edge.
        if abs(hi_a - lo_b) <= EPS_M:
            at = hi_a
        elif abs(hi_b - lo_a) <= EPS_M:
            at = lo_a
        else:
            continue

        lo_c, hi_c = _span(a, axis.other)
        lo_d, hi_d = _span(b, axis.other)
        lo, hi = max(lo_c, lo_d), min(hi_c, hi_d)
        if hi - lo >= min_length:
            return SharedWall(axis=axis, at=at, lo=lo, hi=hi)
    return None


def _span(rect: Rect, axis: Axis) -> tuple[float, float]:
    return (rect.x0, rect.x1) if axis is Axis.X else (rect.y0, rect.y1)


def manhattan(a: tuple[float, float], b: tuple[float, float]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
