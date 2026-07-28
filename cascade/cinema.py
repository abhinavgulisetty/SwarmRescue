"""Making the scenario legible on camera.

This is not decoration.  A rescue swarm inside a collapsed building is, by
construction, a thing you cannot see: the robots are 0.34 m across in a
24 x 18 x 12 m ruin, and they spend the entire run behind concrete.  The first
cut of the Phase 0 demo rendered a chase camera at a fixed azimuth, which put
a wall between the lens and the robot within seconds and kept it there --
several hundred frames of an empty grey wall, with the robot alive and driving
the whole time, invisible.

Three things fix it, and all three are geometric rather than aesthetic:

* **Dollhouse culling.**  Everything above the storey being watched is removed,
  so the camera looks down into the building instead of at the outside of it.
* **Translucent partitions.**  Walls below the cut stay visible enough to read
  the floor plan but no longer hide what is moving behind them.
* **A camera that tracks the swarm** rather than pointing in a fixed direction.

The plan pane complements the hero shot: with every slab removed, a top-down
view shows every robot at once regardless of which storey it is on, which is
the only single frame that answers "where is the swarm".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .render import CameraView

#: Structural member name prefixes, as emitted by `cascade.structure`.
_MEMBER_PREFIXES: tuple[str, ...] = ("col_", "beam_", "slab_", "wall_", "stair_")

#: Alpha for partitions under the cut: solid enough to read the plan, sheer
#: enough to see a robot through.
WALL_ALPHA: float = 0.26
#: Alpha for walls in the plan pane, where they are the only context left.
PLAN_WALL_ALPHA: float = 0.75
#: Height of the cut above the robot being followed, metres.  Clears the
#: robot's beacon and the door heads, and stays under the next slab up.
CEILING_MARGIN_M: float = 1.75
#: How far above or below the subject another robot counts as being on the
#: same storey, and so worth keeping in frame, metres.
COMPANION_HEIGHT_M: float = 1.6


@dataclass(frozen=True, slots=True)
class GeomGroups:
    """Geom ids grouped by what they are, with their original colours."""

    slabs: np.ndarray
    walls: np.ndarray
    frame: np.ndarray
    robots: np.ndarray
    base_rgba: np.ndarray
    #: World height of each geom's lowest corner.  Precomputed because the
    #: structural geoms are static, so it never changes after settling.
    low_z: np.ndarray

    @property
    def structural(self) -> np.ndarray:
        return np.concatenate((self.slabs, self.walls, self.frame))


def index_geoms(model: mujoco.MjModel, data: mujoco.MjData) -> GeomGroups:
    """Classify every geom, snapshot its colour, and measure how low it sits.

    Materials are shared per member kind, so tinting one wall through its
    material would tint all of them.  The colours are copied out here and the
    material link is severed in `_apply`, which buys per-geom control at the
    cost of having to carry the originals.
    """
    slabs: list[int] = []
    walls: list[int] = []
    frame: list[int] = []
    robots: list[int] = []
    base = np.zeros((model.ngeom, 4), dtype=np.float32)
    low_z = np.zeros(model.ngeom, dtype=np.float64)

    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        matid = int(model.geom_matid[geom])
        base[geom] = (
            model.mat_rgba[matid] if matid >= 0 else model.geom_rgba[geom]
        )

        # The half-extent *along world z*, not the largest half-extent.  Using
        # the largest is a subtle and total failure: a floor slab is 6 m across
        # and 0.1 m thick, so it reports its underside 3 m below its centre,
        # never counts as above the cut, and the dollhouse view renders the
        # building with its roof on.
        rot = np.asarray(data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
        half = np.asarray(model.geom_size[geom][:3], dtype=np.float64)
        low_z[geom] = float(data.geom_xpos[geom][2] - np.abs(rot[2, :]) @ half)

        member = name[2:] if name.startswith("g_") else ""
        if member.startswith("slab_"):
            slabs.append(geom)
        elif member.startswith("wall_"):
            walls.append(geom)
        elif any(member.startswith(p) for p in _MEMBER_PREFIXES):
            frame.append(geom)
        elif "robot" in name:
            robots.append(geom)

    def as_array(ids: list[int]) -> np.ndarray:
        return np.asarray(ids, dtype=np.int32)

    return GeomGroups(
        slabs=as_array(slabs),
        walls=as_array(walls),
        frame=as_array(frame),
        robots=as_array(robots),
        base_rgba=base,
        low_z=low_z,
    )


def _apply(model: mujoco.MjModel, groups: GeomGroups, alpha: np.ndarray) -> None:
    """Write per-geom colours, detaching materials so they take effect."""
    rgba = groups.base_rgba.copy()
    rgba[:, 3] = alpha
    structural = groups.structural
    if structural.size:
        model.geom_matid[structural] = -1
        model.geom_rgba[structural] = rgba[structural]


def _cut_above(groups: GeomGroups, ceiling_z: float) -> np.ndarray:
    """Alpha array with everything above `ceiling_z` made invisible."""
    alpha = groups.base_rgba[:, 3].copy()
    structural = groups.structural
    if structural.size:
        above = structural[groups.low_z[structural] > ceiling_z]
        alpha[above] = 0.0
    return alpha


def hero_style(
    model: mujoco.MjModel, groups: GeomGroups, ceiling_z: float
) -> None:
    """Cut the building off above `ceiling_z` and sheer the walls below it."""
    if not groups.structural.size:
        return
    alpha = _cut_above(groups, ceiling_z)
    kept_walls = groups.walls[groups.low_z[groups.walls] <= ceiling_z]
    alpha[kept_walls] = WALL_ALPHA
    _apply(model, groups, alpha)


def plan_style(
    model: mujoco.MjModel, groups: GeomGroups, ceiling_z: float
) -> None:
    """Top-down view: no slabs at all, and nothing above the cut.

    Slabs go first because from directly overhead they are the only thing
    between the camera and the swarm.  The same cut as the hero pane goes with
    them, so the plan reads as one storey rather than four superimposed --
    with every storey's partitions in shot at once the result is a thicket of
    nested rectangles that nobody can locate a robot in.
    """
    if not groups.structural.size:
        return
    alpha = _cut_above(groups, ceiling_z)
    if groups.slabs.size:
        alpha[groups.slabs] = 0.0
    kept_walls = groups.walls[groups.low_z[groups.walls] <= ceiling_z]
    alpha[kept_walls] = PLAN_WALL_ALPHA
    _apply(model, groups, alpha)


# --- cameras ------------------------------------------------------------------


@dataclass
class SwarmCamera:
    """A camera that keeps a moving swarm in frame.

    Both the aim point and the framing distance are low-pass filtered.  Without
    it the shot snaps every time a robot rounds a corner or a straggler
    finishes, which reads as a fault in the simulation rather than a camera
    move.
    """

    footprint_m: tuple[float, float]
    #: Degrees of orbit per rendered frame -- slow enough to read as a survey.
    orbit_rate_deg: float = 0.16
    azimuth: float = 35.0
    #: Steep enough to look *into* the cut building rather than across it.  At
    #: -32 degrees with the swarm bunched on a staircase the camera closed to
    #: its minimum distance and ended up between two partitions, filling the
    #: frame with the near face of a wall.
    elevation: float = -42.0
    smoothing: float = 0.06
    #: Never close inside this.  The cut exposes one storey, and a lens closer
    #: than the room it is looking into sees the wall, not the room.
    min_distance_m: float = 15.0
    #: Never pull back further than this.  A straggler on the far side of the
    #: building would otherwise drag the framing out until every robot in shot
    #: is a couple of pixels across -- the wide shot that shows everything and
    #: reveals nothing.  Better to lose the straggler from frame and keep the
    #: rest legible; the plan pane still accounts for everyone.
    max_distance_m: float = 30.0
    _lookat: np.ndarray | None = field(default=None, repr=False)
    _distance: float | None = field(default=None, repr=False)

    def update(self, subject: np.ndarray, others: np.ndarray) -> CameraView:
        """Advance the orbit and reframe around `subject` and its neighbours.

        One subject rather than the whole swarm, because the swarm is spread
        over four storeys by design and a single frame cannot hold it.  Aiming
        at the centroid of robots on storeys 1 and 4 points the camera at the
        slab between them; worse, the cut has to clear the highest of them, so
        one robot reaching the roof sealed the building and hid the other
        three.  The plan pane is what accounts for everyone.
        """
        width, depth = self.footprint_m
        if subject is None or np.size(subject) == 0:
            target = np.array([0.5 * width, 0.5 * depth, 1.5])
            spread = max(width, depth)
        else:
            target = np.asarray(subject, dtype=np.float64)
            # Robots on the same storey are in shot anyway; frame them too.
            near = (
                others[np.abs(others[:, 2] - target[2]) < COMPANION_HEIGHT_M]
                if np.size(others)
                else np.zeros((0, 3))
            )
            spread = (
                float(np.linalg.norm(near - target, axis=1).max() * 1.8)
                if near.shape[0]
                else 0.0
            )

        want_distance = min(
            self.max_distance_m, max(self.min_distance_m, spread + 6.0)
        )
        if self._lookat is None or self._distance is None:
            self._lookat = target.copy()
            self._distance = want_distance
        else:
            k = self.smoothing
            self._lookat += k * (target - self._lookat)
            self._distance += k * (want_distance - self._distance)

        self.azimuth = (self.azimuth + self.orbit_rate_deg) % 360.0
        return CameraView(
            name="hero",
            lookat=tuple(float(v) for v in self._lookat),
            distance=float(self._distance),
            azimuth=self.azimuth,
            elevation=self.elevation,
        )

    def ceiling(self, subject: np.ndarray) -> float:
        """Height to cut the building at, given where the subject is."""
        if subject is None or np.size(subject) == 0:
            return 1e9
        return float(np.asarray(subject)[2]) + CEILING_MARGIN_M


#: Vertical extent a MuJoCo camera sees per metre of distance, at the default
#: 45 degree field of view: 2 * tan(22.5 deg).
_VIEW_PER_METRE: float = 0.828


def plan_view(
    footprint_m: tuple[float, float],
    total_height_m: float,
    aspect: float = 1.6,
) -> CameraView:
    """Fixed top-down view framed to the footprint.

    Framed rather than guessed.  The old fixed multiple of the longer side put
    the camera 37 m up over a 24 x 18 m building, which left a third of the
    pane empty on every side and shrank each robot to a couple of pixels in
    the one view whose entire job is to say where they are.
    """
    width, depth = footprint_m
    margin = 1.22
    distance = margin * max(depth, width / aspect) / _VIEW_PER_METRE
    return CameraView(
        name="plan",
        lookat=(0.5 * width, 0.5 * depth, 0.5 * total_height_m),
        distance=distance,
        azimuth=90.0,
        elevation=-89.0,
    )


# --- overlay ------------------------------------------------------------------


_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


def _font(size: int):
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


@dataclass(frozen=True, slots=True)
class RobotReadout:
    """One line of the HUD."""

    name: str
    livery: tuple[float, float, float]
    storey: int
    goal_storey: int
    progress: float
    status: str


def draw_hud(
    frame: np.ndarray,
    *,
    title: str,
    subtitle: str,
    readouts: list[RobotReadout],
    elapsed_s: float,
) -> np.ndarray:
    """Return the frame with a title bar above it and a status panel below.

    Around the frame rather than over it.  Drawn as overlays these two bands
    covered the top and bottom third of the plan pane -- which is where the
    building was -- so the view whose job is to show every robot at once was
    showing the middle half of the floor and hiding the rest behind the
    caption describing it.
    """
    from PIL import Image, ImageDraw

    pane = Image.fromarray(frame).convert("RGB")
    width, pane_height = pane.size

    pad = max(12, width // 80)
    title_font = _font(max(16, width // 46))
    body_font = _font(max(12, width // 74))
    small_font = _font(max(11, width // 88))

    band = pad * 2 + title_font.size + small_font.size
    row_h = body_font.size + 10
    panel_h = pad + row_h * len(readouts) + pad // 2

    height = band + pane_height + panel_h
    image = Image.new("RGB", (width, height), (10, 12, 16))
    image.paste(pane, (0, band))
    draw = ImageDraw.Draw(image, "RGBA")

    # Header band.
    draw.text((pad, pad // 2), title, font=title_font, fill=(238, 240, 245))
    draw.text(
        (pad, pad // 2 + title_font.size + 2),
        subtitle,
        font=small_font,
        fill=(150, 158, 172),
    )
    clock = f"t = {elapsed_s:5.1f} s"
    draw.text(
        (width - pad - draw.textlength(clock, font=body_font), pad),
        clock,
        font=body_font,
        fill=(190, 196, 208),
    )

    # Per-robot rows.
    top = band + pane_height

    bar_w = max(90, width // 7)
    for index, readout in enumerate(readouts):
        y = top + pad // 2 + index * row_h
        colour = tuple(int(255 * c) for c in readout.livery)

        draw.rectangle([pad, y + 3, pad + 12, y + 15], fill=(*colour, 255))
        draw.text(
            (pad + 20, y),
            f"{readout.name}",
            font=body_font,
            fill=(226, 230, 238),
        )
        draw.text(
            (pad + 20 + width // 12, y),
            f"storey {readout.storey} -> {readout.goal_storey}",
            font=body_font,
            fill=(186, 192, 204),
        )

        bar_x = pad + 20 + width // 12 + width // 8
        draw.rectangle(
            [bar_x, y + 4, bar_x + bar_w, y + 14], fill=(44, 48, 58, 255)
        )
        filled = int(bar_w * max(0.0, min(1.0, readout.progress)))
        if filled:
            draw.rectangle(
                [bar_x, y + 4, bar_x + filled, y + 14], fill=(*colour, 255)
            )

        status_colour = {
            "arrived": (140, 220, 150),
            "stalled": (232, 150, 140),
        }.get(readout.status, (176, 182, 196))
        draw.text(
            (bar_x + bar_w + 14, y),
            readout.status,
            font=body_font,
            fill=status_colour,
        )

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
    gap = np.zeros((target_h, 4, 3), dtype=np.uint8)
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
        self._writer = iio.get_writer(
            self.path, fps=fps, codec="libx264", quality=8
        )
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
