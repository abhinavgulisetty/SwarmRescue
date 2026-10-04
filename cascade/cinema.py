"""Making the scenario legible on camera.

This is not decoration.  A rescue swarm inside a collapsed building is, by
construction, a thing you cannot see: the robots are 0.34 m across in a
24 x 18 x 12 m ruin, and they spend the entire run behind concrete.  The first
cut of the Phase 0 demo rendered a chase camera at a fixed azimuth, which put
a wall between the lens and the robot within seconds and kept it there --
several hundred frames of an empty grey wall, with the robot alive and driving
the whole time, invisible.

Four things fix it, and all four are geometric rather than aesthetic:

* **Dollhouse culling.**  Everything above the storey being watched is removed,
  so the camera looks down into the building instead of at the outside of it.
* **Translucent partitions.**  Walls below the cut stay visible enough to read
  the floor plan but no longer hide what is moving behind them.
* **A camera that tracks the swarm** rather than pointing in a fixed direction.
* **Routes and trails drawn into the scene.**  A robot two hundred pixels away
  is a dot; the line it has drawn behind it is legible at any distance, and it
  is also the evidence -- the planned route and the driven track laid over each
  other are the demo's entire claim, in a form a viewer can check.

The plan pane complements the hero shot with a top-down slice of the storey
being watched, which is where the robots on it actually are relative to the
walls around them.

Everything here is about the three-dimensional scene.  The captions and
telemetry drawn on the finished pixels live in `cascade.hud`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise

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
#: Event shots retain the whole frame but make broad occluding surfaces almost
#: transparent, so the viewer sees columns and beams fail instead of a slab.
EVENT_SLAB_ALPHA: float = 0.06
EVENT_WALL_ALPHA: float = 0.10
#: Height of the cut above the robot being followed, metres.  Clears the
#: robot's beacon and the door heads, and stays under the next slab up.
CEILING_MARGIN_M: float = 1.75
#: How far *below* the subject the plan pane's slice starts, metres.  Small and
#: positive: a wall on the storey below ends exactly at this storey's floor, so
#: a slice that reaches even slightly under the robot's wheels picks up the
#: partitions of the floor beneath and the pane fills with a second plan
#: superimposed on the first.
PLAN_FLOOR_DROP_M: float = 0.05
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
    #: World height of each geom's lowest and highest point.  Precomputed
    #: because the structural geoms are static, so neither changes after
    #: settling.
    low_z: np.ndarray
    high_z: np.ndarray

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
    high_z = np.zeros(model.ngeom, dtype=np.float64)

    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        matid = int(model.geom_matid[geom])
        base[geom] = model.mat_rgba[matid] if matid >= 0 else model.geom_rgba[geom]

        # The half-extent *along world z*, not the largest half-extent.  Using
        # the largest is a subtle and total failure: a floor slab is 6 m across
        # and 0.1 m thick, so it reports its underside 3 m below its centre,
        # never counts as above the cut, and the dollhouse view renders the
        # building with its roof on.
        rot = np.asarray(data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
        half = np.asarray(model.geom_size[geom][:3], dtype=np.float64)
        reach = float(np.abs(rot[2, :]) @ half)
        low_z[geom] = float(data.geom_xpos[geom][2]) - reach
        high_z[geom] = float(data.geom_xpos[geom][2]) + reach

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
        high_z=high_z,
    )


def _apply(model: mujoco.MjModel, groups: GeomGroups, alpha: np.ndarray) -> None:
    """Write per-geom colours, detaching materials so they take effect."""
    rgba = groups.base_rgba.copy()
    rgba[:, 3] = alpha
    styled = np.concatenate((groups.structural, groups.robots))
    if styled.size:
        model.geom_matid[styled] = -1
        model.geom_rgba[styled] = rgba[styled]


def _cut_above(groups: GeomGroups, ceiling_z: float) -> np.ndarray:
    """Alpha array with everything above `ceiling_z` made invisible."""
    alpha = groups.base_rgba[:, 3].copy()
    structural = groups.structural
    if structural.size:
        above = structural[groups.low_z[structural] > ceiling_z]
        alpha[above] = 0.0
    return alpha


def hero_style(model: mujoco.MjModel, groups: GeomGroups, ceiling_z: float) -> None:
    """Cut the building off above `ceiling_z` and sheer the walls below it."""
    if not groups.structural.size:
        return
    alpha = _cut_above(groups, ceiling_z)
    kept_walls = groups.walls[groups.low_z[groups.walls] <= ceiling_z]
    alpha[kept_walls] = WALL_ALPHA
    _apply(model, groups, alpha)


def event_style(model: mujoco.MjModel, groups: GeomGroups) -> None:
    """Expose the structural frame for a short collapse event shot."""

    if not groups.structural.size:
        return
    alpha = groups.base_rgba[:, 3].copy()
    alpha[groups.slabs] = EVENT_SLAB_ALPHA
    alpha[groups.walls] = EVENT_WALL_ALPHA
    _apply(model, groups, alpha)


def highlight_members(model: mujoco.MjModel, names: set[str]) -> None:
    """Tint failed structural members so the released weld is visible."""

    for name in names:
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"g_{name}")
        if geom >= 0:
            model.geom_matid[geom] = -1
            model.geom_rgba[geom] = (1.0, 0.08, 0.02, 1.0)


def plan_style(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    groups: GeomGroups,
    floor_z: float,
    ceiling_z: float,
) -> None:
    """Top-down view of the one storey between `floor_z` and `ceiling_z`.

    Slabs go first: from directly overhead they are the only thing between the
    camera and the swarm.

    Cutting only from above -- the first version -- is not enough, and the
    result was the single worst frame in the demo.  Watching a robot on the
    third storey, everything from the ground up was still in shot, so the pane
    showed four floor plans stacked in projection: forty-odd partitions of
    identical colour, none of them at the height anything was happening, and a
    robot somewhere inside as a dot.  A floor plan is only a floor plan if it
    is of one floor.
    """
    if not groups.structural.size:
        return
    alpha = np.zeros_like(groups.base_rgba[:, 3])
    structural = groups.structural
    inside = structural[
        (groups.high_z[structural] > floor_z) & (groups.low_z[structural] < ceiling_z)
    ]
    alpha[inside] = groups.base_rgba[inside, 3]
    if groups.slabs.size:
        alpha[groups.slabs] = 0.0
    kept_walls = np.intersect1d(inside, groups.walls, assume_unique=False)
    alpha[kept_walls] = PLAN_WALL_ALPHA
    # Robot geometry is dynamic, so the structural bounds cached by
    # `index_geoms` cannot decide which storey it occupies.  Use the root body
    # position from this frame and hide markers on every other floor.  Without
    # this, a plan labelled "storey 2" showed four floating lollipops belonging
    # to four different heights.
    for geom in groups.robots:
        body = int(model.geom_bodyid[geom])
        body_z = float(data.xpos[body][2])
        if floor_z <= body_z < ceiling_z:
            alpha[geom] = groups.base_rgba[geom, 3]
    _apply(model, groups, alpha)


# --- routes and trails --------------------------------------------------------

#: Radius of the drawn planned route, metres.
ROUTE_WIDTH_M: float = 0.035
#: ...and of the track the robot has actually left, drawn thicker because it is
#: the claim the demo is making.  The route is what was promised; the trail is
#: what happened, and the gap between them is the only thing worth looking at.
TRAIL_WIDTH_M: float = 0.06
#: How far the lines float above the surface they lie on, metres.  Enough to
#: clear z-fighting with a floor slab at any camera distance in use.
LINE_LIFT_M: float = 0.07
#: Route waypoints per drawn vertex.  The planner emits one per 0.25 m voxel,
#: and a capsule per voxel is thousands of geoms for a line the eye reads as
#: smooth at a quarter of that.
ROUTE_STRIDE: int = 4
#: Alpha of the planned route.  Present but recessive: it is context for the
#: trail, not the subject.
ROUTE_ALPHA: float = 0.40
#: Destination beacons float above the floor so route endpoints stay obvious
#: even behind low rubble in the dollhouse view.
GOAL_MARKER_HEIGHT_M: float = 0.62
GOAL_MARKER_RADIUS_M: float = 0.16


@dataclass(frozen=True, slots=True)
class PathTrace:
    """One robot's planned route and the track it has actually left."""

    livery: tuple[float, float, float]
    route: np.ndarray
    trail: np.ndarray
    rejected_route: np.ndarray | None = None


@dataclass
class Trails:
    """Where each robot has been, sampled once per rendered frame.

    Bounded in both length and density on purpose.  A full traversal is well
    over a thousand frames, and a robot that stalls contributes a frame's worth
    of trail for every frame it spends not moving -- so an unbounded buffer is
    mostly a record of standing still, and it is the robot that got furthest
    whose history gets dropped when the renderer runs out of geoms.
    """

    n_robots: int
    max_points: int = 300
    #: Minimum distance between stored points, metres.
    min_step_m: float = 0.12
    _points: list[list[np.ndarray]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._points = [[] for _ in range(self.n_robots)]

    def record(self, positions: np.ndarray) -> None:
        for index, position in enumerate(np.asarray(positions, dtype=np.float64)):
            track = self._points[index]
            if track and float(np.linalg.norm(position - track[-1])) < self.min_step_m:
                continue
            track.append(position.copy())
            if len(track) > self.max_points:
                # Thin the oldest half rather than dropping the head, so the
                # whole journey stays on screen at declining resolution.  The
                # slice starts at 1 so that the very first point -- the door
                # the robot came in by -- is never the one discarded.
                del track[1 : len(track) // 2 : 2]

    def of(self, index: int) -> np.ndarray:
        track = self._points[index]
        return np.asarray(track, dtype=np.float64) if track else np.zeros((0, 3))


def _connect(scene, start, end, rgba, width: float) -> bool:
    """Append one capsule spanning `start` to `end`.  False if the buffer is full."""
    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        width,
        np.ascontiguousarray(start, dtype=np.float64),
        np.ascontiguousarray(end, dtype=np.float64),
    )
    scene.ngeom += 1
    return True


def _polyline(
    scene,
    points: np.ndarray,
    rgba,
    width: float,
    z_range: tuple[float, float] | None = None,
) -> None:
    lift = np.array([0.0, 0.0, LINE_LIFT_M])
    for raw_start, raw_end in pairwise(points):
        if z_range is not None:
            low, high = z_range
            # Do not join two visible fragments across an excluded storey.  A
            # segment belongs to the slice only when both endpoints do.
            if not (low <= raw_start[2] < high and low <= raw_end[2] < high):
                continue
        start, end = raw_start + lift, raw_end + lift
        if not _connect(scene, start, end, rgba, width):
            return


def _dashed_polyline(
    scene,
    points: np.ndarray,
    rgba,
    width: float,
    z_range: tuple[float, float] | None = None,
) -> None:
    """Draw alternating segments so a rejected route remains unambiguous."""

    vertices = _route_vertices(points)
    lift = np.array([0.0, 0.0, LINE_LIFT_M * 1.35])
    for index, (raw_start, raw_end) in enumerate(pairwise(vertices)):
        if index % 2:
            continue
        if z_range is not None:
            low, high = z_range
            if not (low <= raw_start[2] < high and low <= raw_end[2] < high):
                continue
        if not _connect(scene, raw_start + lift, raw_end + lift, rgba, width):
            return


def _draw_goal_marker(
    scene,
    point: np.ndarray,
    colour: tuple[float, float, float],
    z_range: tuple[float, float] | None,
) -> None:
    """Draw a colour-matched mast and beacon at one route destination."""

    if z_range is not None and not (z_range[0] <= point[2] < z_range[1]):
        return
    base = np.asarray(point, dtype=np.float64).copy()
    base[2] += LINE_LIFT_M
    top = base.copy()
    top[2] += GOAL_MARKER_HEIGHT_M
    if not _connect(scene, base, top, (*colour, 0.9), ROUTE_WIDTH_M * 1.2):
        return
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, GOAL_MARKER_RADIUS_M),
        top,
        np.eye(3).ravel(),
        np.asarray((*colour, 0.82), dtype=np.float32),
    )
    scene.ngeom += 1


def draw_paths(
    scene,
    traces: list[PathTrace],
    *,
    z_range: tuple[float, float] | None = None,
) -> None:
    """Overlay every robot's planned route and driven trail onto `scene`.

    This is what turns the demo from footage of things moving into a claim
    anyone can check: the thin line is where the planner said the robot could
    go, the thick one is where it went, and a viewer can see for themselves
    whether the second follows the first.
    """
    for trace in traces:
        colour = tuple(trace.livery)
        if trace.rejected_route is not None and trace.rejected_route.shape[0] > 1:
            _dashed_polyline(
                scene,
                trace.rejected_route,
                (1.0, 0.10, 0.06, 0.9),
                ROUTE_WIDTH_M * 1.35,
                z_range,
            )
        if trace.route.shape[0] > 1:
            route_colour = (
                (0.16, 0.92, 0.42) if trace.rejected_route is not None else colour
            )
            _polyline(
                scene,
                _route_vertices(trace.route),
                (
                    *route_colour,
                    0.9 if trace.rejected_route is not None else ROUTE_ALPHA,
                ),
                ROUTE_WIDTH_M * (1.35 if trace.rejected_route is not None else 1.0),
                z_range,
            )
        if trace.trail.shape[0] > 1:
            _polyline(
                scene,
                trace.trail,
                (*colour, 1.0),
                TRAIL_WIDTH_M,
                z_range,
            )
        if trace.route.shape[0] > 0:
            _draw_goal_marker(
                scene,
                trace.route[-1],
                colour,
                z_range,
            )


def _route_vertices(points: np.ndarray) -> np.ndarray:
    """Thin a route without erasing its turns or final goal.

    Blind stride sampling draws a chord across any corner that happens to
    fall between samples and commonly drops the final waypoint.  That makes
    the visual evidence contradict the controller's real route precisely at
    the places a viewer most needs to inspect.  Keep the regular samples for
    long straight runs, plus every change of direction and both endpoints.
    """
    route = np.asarray(points, dtype=np.float64)
    if route.shape[0] <= 2:
        return route
    keep = {0, route.shape[0] - 1}
    keep.update(range(0, route.shape[0], ROUTE_STRIDE))
    segments = np.diff(route, axis=0)
    length = np.linalg.norm(segments, axis=1)
    for index in range(1, route.shape[0] - 1):
        before = segments[index - 1]
        after = segments[index]
        if length[index - 1] < 1e-9 or length[index] < 1e-9:
            keep.add(index)
            continue
        alignment = float(
            np.clip(
                np.dot(before, after) / (length[index - 1] * length[index]),
                -1.0,
                1.0,
            )
        )
        if alignment < 1.0 - 1e-6:
            keep.add(index)
    return route[np.asarray(sorted(keep), dtype=np.int64)]


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

    def reset(self) -> None:
        """Start a clean composition after an intentional subject cut."""
        self._lookat = None
        self._distance = None

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

        want_distance = min(self.max_distance_m, max(self.min_distance_m, spread + 6.0))
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


#: Framing margin around the footprint in the plan pane.
_PLAN_MARGIN: float = 1.22


def plan_view(
    footprint_m: tuple[float, float],
    floor_z: float,
    aspect: float = 1.6,
) -> CameraView:
    """Top-down view of the storey at `floor_z`, framed to the footprint.

    Framed rather than guessed.  The old fixed multiple of the longer side put
    the camera 37 m up over a 24 x 18 m building, which left a third of the
    pane empty on every side and shrank each robot to a couple of pixels in
    the one view whose entire job is to say where they are.

    Aimed at the storey on show rather than at the middle of the building, so
    that the plan is the same size whichever storey it is of.  Under a fixed
    aim point the ground floor sat 33 m from the lens and the roof 21 m, and
    the pane silently changed scale by half as the swarm climbed.
    """
    width, depth = footprint_m
    distance = _PLAN_MARGIN * max(depth, width / aspect) / _VIEW_PER_METRE
    return CameraView(
        name="plan",
        lookat=(0.5 * width, 0.5 * depth, float(floor_z)),
        distance=distance,
        azimuth=90.0,
        elevation=-89.0,
    )
