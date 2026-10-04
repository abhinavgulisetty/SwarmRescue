"""Stage 4 -- dual representation extraction.

The settled scene is rasterized into an anisotropic voxel grid and reduced to
a **traversability graph**: the cells a robot can stand on, and the moves it
can make between them.

The grid is deliberately anisotropic -- 0.4 m horizontally, 0.1 m vertically.
A 0.4 m cube cannot represent whether a 0.18 m step is climbable, so a cubic
grid at the horizontal resolution would answer the one question the graph
exists to answer by rounding it away.

`G` is ground truth for evaluation and for the offline baselines.  It is never
handed to the robots: they only ever see the noisy local samples of SS4.4.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .params import TraversabilityParams

#: Padding around the building footprint, metres, so debris thrown clear of
#: the structure is still inside the grid.
GRID_MARGIN_M: float = 1.5

#: How far a box's local axis may tilt out of horizontal and still be grown by
#: the configuration-space pad, as |world z component|.  About 8.6 degrees.
_HORIZONTAL_TOL: float = 0.15

#: Geom name prefix for members that are running surfaces rather than
#: obstacles, and so are never grown into configuration-space obstacles.
_WALKABLE_PREFIX: str = "g_stair_"

#: Straight down, for the surface probes.
_DOWN: np.ndarray = np.array([0.0, 0.0, -1.0], dtype=np.float64)

#: What a full-height step adds to the cost of a move, in units of the cost of
#: a flat one.  Deliberately steep -- a detour has to be sixty cells long
#: before mounting a full step wins -- because the planner's shortcuts are
#: exactly the manoeuvres the platform cannot execute.  With this at 25 the
#: route to the stairs flipped between climbing the side of the entrance ramp
#: and driving round to its foot depending on which was a cell shorter, and
#: three robots in four wedged themselves on the side.
_STEP_PENALTY: float = 60.0

#: Where to check that the ground still reaches, in units of the robot radius,
#: around a cell the robot's centre might occupy.  Four points rather than a
#: ring: the chassis is roughly square and the failure being ruled out -- one
#: side of the robot hanging over the edge of a stair flight -- shows up on an
#: axis long before it shows up on a diagonal.
_FOOTPRINT_PROBES: tuple[tuple[float, float], ...] = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
)


@dataclass(frozen=True, slots=True)
class VoxelGrid:
    """Occupancy over an axis-aligned lattice.

    Two occupancies, not one.  `occupied` is where matter is, and answers
    "what can I stand on".  `blocked` is the same geometry grown by the
    robot's turning radius, and answers "can the robot's *centre* be here" --
    the configuration-space obstacle.  Keeping both is what lets a cell be
    simultaneously solid ground to stand on and, one level up, too close to a
    wall for the chassis to occupy.
    """

    origin: tuple[float, float, float]
    cell: tuple[float, float, float]
    occupied: np.ndarray
    blocked: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.occupied.shape  # type: ignore[return-value]

    def center(self, i: int, j: int, k: int) -> tuple[float, float, float]:
        return (
            self.origin[0] + (i + 0.5) * self.cell[0],
            self.origin[1] + (j + 0.5) * self.cell[1],
            self.origin[2] + (k + 0.5) * self.cell[2],
        )

    def surface_z(self, k: int) -> float:
        """World height of the top face of level `k`."""
        return self.origin[2] + (k + 1) * self.cell[2]

    def occupancy_fraction(self) -> float:
        return float(self.occupied.mean())


@dataclass(frozen=True, slots=True)
class TraversabilityGraph:
    """Standable cells and the moves between them.

    `cells[n] = (i, j, k)` indexes the grid; `surface_z[n]` is the height the
    robot's wheels sit at.  Edges are stored as parallel arrays so the whole
    thing is cheap to serialise alongside a run manifest.
    """

    grid: VoxelGrid
    cells: np.ndarray
    surface_z: np.ndarray
    edges: np.ndarray
    edge_cost: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.cells.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    def world_positions(self) -> np.ndarray:
        """(N, 3) world coordinates of each standable cell's surface."""
        out = np.empty((self.n_nodes, 3), dtype=np.float64)
        cx, cy, _ = self.grid.cell
        ox, oy, _ = self.grid.origin
        out[:, 0] = ox + (self.cells[:, 0] + 0.5) * cx
        out[:, 1] = oy + (self.cells[:, 1] + 0.5) * cy
        out[:, 2] = self.surface_z
        return out

    def adjacency(self) -> list[list[tuple[int, float]]]:
        adj: list[list[tuple[int, float]]] = [[] for _ in range(self.n_nodes)]
        for (a, b), cost in zip(self.edges, self.edge_cost, strict=True):
            adj[int(a)].append((int(b), float(cost)))
            adj[int(b)].append((int(a), float(cost)))
        return adj


# --- rasterization -----------------------------------------------------------


def rasterize(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    footprint_m: tuple[float, float],
    total_height_m: float,
    params: TraversabilityParams,
    *,
    margin_m: float = GRID_MARGIN_M,
) -> VoxelGrid:
    """Voxelize every colliding box geom at its current pose.

    Boxes are rasterized as oriented boxes, not as their axis-aligned bounds:
    after a collapse debris is rotated, and an AABB of a tilted slab would
    seal voids that the robot can actually drive through -- exactly the
    features that make a rubble field interesting.
    """
    width, depth = footprint_m
    if not math.isfinite(margin_m) or margin_m < 0:
        raise ValueError("horizontal map margin must be finite and nonnegative")
    cell = (params.voxel_size_m, params.voxel_size_m, params.voxel_height_m)
    # The grid starts one level *below* z = 0 so that the ground itself can be
    # represented as a solid layer.  Without it there is no floor at storey 0:
    # the ground is a plane geom, this rasterizer only handles boxes, and the
    # lowest standable surface in the whole building came out at z = 3.0 --
    # the top of the first slab.
    origin = (-margin_m, -margin_m, -cell[2])
    shape = (
        math.ceil((width + 2 * margin_m) / cell[0]),
        math.ceil((depth + 2 * margin_m) / cell[1]),
        math.ceil((total_height_m + GRID_MARGIN_M) / cell[2]) + 1,
    )
    occupied = np.zeros(shape, dtype=bool)
    # Level 0 spans [-cell_z, 0], so its top face is exactly the ground plane.
    occupied[:, :, 0] = True
    blocked = np.zeros(shape, dtype=bool)
    blocked[:, :, 0] = True

    # The two grids round in *opposite* directions, and they have to.
    #
    # `blocked` answers "would the chassis hit something here", so it must
    # never miss an obstacle: every box is grown by the robot's radius, which
    # is larger than half a voxel and so also guarantees a 0.15 m partition
    # cannot slip between sample points.  It is not grown vertically, since a
    # vertical pad would lift every floor slab into the level above it.
    #
    # `occupied` answers "is there something solid under the tracks", so it
    # must never invent material *laterally*: half a voxel of phantom floor
    # past the edge of a 34 degree stair flight reads as a standable surface
    # 0.2 m up in mid-air, and the planner routed robots onto it.  They drove
    # at a floor that was not there, dropped, and wedged against the real edge
    # of the flight.
    #
    # Vertically it is padded by half a level, which invents nothing: it is
    # the rounding the height lattice already imposes, and without it a stair
    # flight thin enough to fall between two level centres vanishes from the
    # support grid entirely and disconnects the storeys above it.
    support_pad = np.array([0.0, 0.0, 0.5 * cell[2]])
    robot_pad = np.array([params.robot_radius_m, params.robot_radius_m, 0.0])

    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        # Pulverized members have their collision disabled; they are gone.
        if model.geom_contype[geom] == 0 and model.geom_conaffinity[geom] == 0:
            continue

        half = np.asarray(model.geom_size[geom][:3], dtype=np.float64)
        center = np.asarray(data.geom_xpos[geom], dtype=np.float64)
        rot = np.asarray(data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
        _mark_box(occupied, origin, cell, center, rot, half, support_pad)

        # Stairs still occupy space -- you cannot stand in the 0.2 m gap under
        # a flight -- but they are not grown, because a member the robot drives
        # *on* is not an obstacle to keep clear of.  Growing them laterally
        # hangs a phantom overhang past the edge of every landing, and a
        # landing overhanging the flight below it removes exactly the headroom
        # a robot needs to step off the stairs onto the floor.  That severed
        # the storeys above the first while the surfaces themselves were all
        # present and correctly placed.
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        pad = support_pad if name.startswith(_WALKABLE_PREFIX) else robot_pad
        _mark_box(blocked, origin, cell, center, rot, half, pad, horizontal_only=True)

    return VoxelGrid(origin=origin, cell=cell, occupied=occupied, blocked=blocked)


def _mark_box(
    occupied: np.ndarray,
    origin: tuple[float, float, float],
    cell: tuple[float, float, float],
    center: np.ndarray,
    rot: np.ndarray,
    half: np.ndarray,
    pad: np.ndarray,
    *,
    horizontal_only: bool = False,
) -> None:
    """Set every voxel whose centre lies within `pad` of the oriented box.

    Testing whether the voxel *centre* lies inside the box is not enough: a
    0.15 m partition inside a 0.4 m voxel only marks it occupied when it
    happens to cover the centre, so most walls in the building were invisible
    to the raster and the planner routed straight through them.  The robot
    then drove into a wall the graph believed was open floor and stalled.

    Instead the box is inflated by `pad` projected onto each of its own axes,
    which is the standard conservative AABB-vs-OBB overlap test.  It never
    misses an overlap; it can over-report slightly at corners, and
    over-reporting is the safe direction for a traversability map.

    Passing the robot's radius as `pad` turns the same routine into a
    configuration-space obstacle: the inflated box is exactly the set of
    centre positions at which the robot would be in collision.

    `horizontal_only` pads a local axis only if that axis is genuinely
    horizontal.  Growing a box along an axis that tilts out of the horizontal
    plane moves its surfaces vertically, and for a member a robot drives *on*
    that invents obstacles out of thin air:

    * Along the thickness of a 34 degree stair flight, r of growth lifts its
      top face by r*sin(34)*cos(34) -- a whole voxel level -- so the flight
      marks the space directly above itself as occupied and nothing can stand
      on a staircase.
    * Along the *run* of the same flight, r of growth extends the ramp past
      its own top end and up into the air, leaving a phantom slab hovering
      over the landing at the head of the stairs.  That one is worse, because
      it lands precisely where a robot must be to step off the stairs onto the
      floor, and it severed every storey above the first.

    For an upright wall or column the vertical axis is the height, for a slab
    or a flight it is the thickness, and for a ramp the run tilts too.  In
    every case the axes that survive are the ones along which growth means
    "keep clear laterally", which is the only thing configuration space is
    being asked for here.
    """
    # The world-frame pad has to be resolved onto the box's own axes before it
    # can be dropped selectively -- the axes to spare are *local* ones.
    local_pad = np.abs(rot).T @ pad
    if horizontal_only:
        local_pad = np.where(np.abs(rot[2, :]) > _HORIZONTAL_TOL, 0.0, local_pad)
    inflated = half + local_pad
    extent = np.abs(rot) @ half + pad
    lo = center - extent
    hi = center + extent

    start = [max(0, math.floor((lo[d] - origin[d]) / cell[d])) for d in range(3)]
    stop = [
        min(occupied.shape[d], math.ceil((hi[d] - origin[d]) / cell[d]) + 1)
        for d in range(3)
    ]
    if any(stop[d] <= start[d] for d in range(3)):
        return

    axes = [
        origin[d] + (np.arange(start[d], stop[d]) + 0.5) * cell[d] for d in range(3)
    ]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    points = np.stack((gx, gy, gz), axis=-1) - center
    # Into the box frame, where the containment test is a box comparison.
    local = points @ rot
    inside = np.all(np.abs(local) <= inflated, axis=-1)
    if inside.any():
        occupied[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]] |= inside


def height_map(grid: VoxelGrid) -> np.ndarray:
    """Top occupied surface height per column, for heightfield-style plots.

    Columns with nothing in them read as the grid floor.  This is a *display*
    product: the robot never sees it, and the graph is built from the full 3-D
    occupancy rather than from this, because a single height per column cannot
    represent a void under a leaning slab.
    """
    nx, ny, nz = grid.shape
    any_occupied = grid.occupied.any(axis=2)
    topmost = nz - 1 - np.argmax(grid.occupied[:, :, ::-1], axis=2)
    heights = np.where(
        any_occupied, grid.origin[2] + (topmost + 1) * grid.cell[2], grid.origin[2]
    )
    return heights.reshape(nx, ny)


# --- the traversability graph ------------------------------------------------


def build_graph(
    grid: VoxelGrid,
    params: TraversabilityParams,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> TraversabilityGraph:
    """Reduce occupancy to standable cells and the moves between them.

    The scene comes back in as well as the grid, because the lattice on its own
    cannot answer the question that decides whether a route is drivable.  A
    0.1 m height lattice sees a 34 degree ramp as a 0.17 m step per cell and a
    0.17 m cliff as the same thing, so any threshold that lets the swarm up a
    staircase also lets it plan straight over a ledge.  Heights and the
    discontinuity between them are measured against the real geometry instead.
    """
    cells, surface = _standable_cells(grid, params)
    if cells.shape[0] == 0:
        empty_i = np.zeros((0, 2), dtype=np.int32)
        return TraversabilityGraph(
            grid=grid,
            cells=cells,
            surface_z=surface,
            edges=empty_i,
            edge_cost=np.zeros(0, dtype=np.float64),
        )

    surface, drop = _probe_surfaces(grid, cells, surface, params, model, data)
    supported = np.isfinite(drop).all(axis=1)
    cells, surface, drop = cells[supported], surface[supported], drop[supported]
    edges, costs = _connect(cells, surface, drop, grid, params, model, data)
    return TraversabilityGraph(
        grid=grid,
        cells=cells,
        surface_z=surface,
        edges=edges,
        edge_cost=costs,
    )


def _standable_cells(
    grid: VoxelGrid, params: TraversabilityParams
) -> tuple[np.ndarray, np.ndarray]:
    """Cells with solid beneath and enough headroom above.

    A column can yield several: a floor, a slab above it, and the top of a
    debris pile in between are all places a robot might stand.  That is why
    the graph is built in 3-D rather than from a height map.
    """
    occupied = grid.occupied
    # Support is a question about matter -- is there something under the
    # tracks -- so it reads the true occupancy.  Everything above the surface
    # is a question about whether the *chassis* fits, so it reads the
    # configuration-space grid.  Using `occupied` for both is what let the
    # planner thread a 0.42 m-wide robot through a 0.25 m slot.
    blocked = grid.blocked
    _, _, nz = grid.shape
    need = params.clearance_levels

    solid = occupied[:, :, :-1]
    free_above = ~blocked[:, :, 1:]
    candidate = solid & free_above

    # Require `need` consecutive free levels above the surface.
    headroom = np.ones_like(candidate)
    for offset in range(1, need + 1):
        shifted = np.zeros_like(candidate)
        upper = nz - 1 - offset
        if upper > 0:
            shifted[:, :, :upper] = ~blocked[:, :, 1 + offset : nz]
        headroom &= shifted
    candidate &= headroom

    idx = np.argwhere(candidate)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.int32), np.zeros(0, dtype=np.float64)

    surface = grid.origin[2] + (idx[:, 2] + 1) * grid.cell[2]
    return idx.astype(np.int32), surface.astype(np.float64)


def _drop(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    x: float,
    y: float,
    ceiling: float,
    span: float,
) -> float | None:
    """Height of the first surface below `(x, y, ceiling)`, within `span`."""
    hit = np.zeros(1, dtype=np.int32)
    distance = mujoco.mj_ray(
        model,
        data,
        np.array([x, y, ceiling], dtype=np.float64),
        _DOWN,
        None,
        1,  # include static geoms: the ground plane and the standing frame
        -1,
        hit,
    )
    if distance < 0.0 or distance > span:
        return None
    return ceiling - float(distance)


def _probe_surfaces(
    grid: VoxelGrid,
    cells: np.ndarray,
    lattice_z: np.ndarray,
    params: TraversabilityParams,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray]:
    """Real surface height under each cell, and how far the ground falls away.

    Two things the lattice cannot supply.

    The height, because the lattice rounds to 0.1 m, which is enough to put a
    waypoint 0.1 m inside a stair tread or 0.1 m above it -- and because
    rounding destroys the very signal `_connect` needs: on a 0.1 m lattice a
    smooth ramp and a flight of ledges are the same sequence of numbers.

    The drops, because occupancy answers "is there material under this point"
    and a robot is not a point.  A cell along the edge of a stair flight has
    material under its centre and nothing under half the chassis, and every
    robot the planner sent up such a cell dropped a wheel over the side and
    hung there.  Configuration space already keeps the chassis clear of what
    it would *hit*; this measures what holds it up.

    Returns the depths in `_FOOTPRINT_PROBES` order, as positive distances
    below the cell's own surface, and infinite where there is no ground within
    reach at all.  What counts as too deep is left to `_connect`, because it
    depends on which way the robot is going: the drop off the downhill end of
    a ramp is the ramp, and the same drop off its side is a fall.
    """
    ox, oy, oz = grid.origin
    cx, cy, cz = grid.cell
    radius = params.robot_radius_m
    # Deepest drop still worth measuring: a full legal grade over the probe
    # distance, plus a full step.  Nothing beyond that is drivable in any
    # direction, so the cell is discarded outright.
    reach = radius * math.tan(math.radians(params.max_slope_deg))
    reach += params.max_step_height_m
    height = np.array(lattice_z, dtype=np.float64, copy=True)
    drop = np.zeros((cells.shape[0], len(_FOOTPRINT_PROBES)), dtype=np.float64)

    for node in range(cells.shape[0]):
        i, j, k = (int(v) for v in cells[node])
        x = ox + (i + 0.5) * cx
        y = oy + (j + 0.5) * cy
        # Start one level up -- the cell is known to be clear there -- and
        # allow the ray to fall through the level whose material supports it.
        found = _drop(model, data, x, y, oz + (k + 2) * cz, 2.05 * cz)
        if found is not None:
            height[node] = found
        under = height[node]

        for probe, (dx, dy) in enumerate(_FOOTPRINT_PROBES):
            # Cast from above, because the uphill side of a ramp is *higher*
            # than the cell centre and a ray launched level with it would sail
            # straight under the ramp and report the void beneath instead.
            found = _drop(
                model,
                data,
                x + dx * radius,
                y + dy * radius,
                under + reach + 0.5 * cz,
                2.0 * reach + cz,
            )
            drop[node, probe] = math.inf if found is None else under - found
    return height, drop


def _connect(
    cells: np.ndarray,
    surface: np.ndarray,
    drop: np.ndarray,
    grid: VoxelGrid,
    params: TraversabilityParams,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> tuple[np.ndarray, np.ndarray]:
    """Join horizontally adjacent standable cells the robot can drive between.

    Every move is split into the part the robot rolls up -- a grade -- and the
    part it has to *mount* -- a step -- because the platform's limits on the
    two differ by a factor of two and they are not interchangeable.  The split
    comes from sampling the surface halfway between the cells: on a ramp the
    midpoint sits halfway up, and on a ledge it sits at one end or the other.

    Without the split the graph cannot express a staircase at all.  A 34
    degree flight rises 0.17 m per 0.25 m cell, so any rise threshold generous
    enough to climb it also admits a 0.17 m vertical drop -- and the planner
    used that to route four robots up the *side* of a stair threshold instead
    of along it.  All four wedged against a 0.13 m lateral edge, two metres
    from an entrance ramp that would have taken them straight up.
    """
    lookup: dict[tuple[int, int], list[int]] = {}
    for node, (i, j, _) in enumerate(cells):
        lookup.setdefault((int(i), int(j)), []).append(node)

    ox, oy, _ = grid.origin
    cx, cy, cz = grid.cell
    grade_cap = cx * math.tan(math.radians(params.max_slope_deg))
    step_cap = params.max_step_height_m
    # Nothing further apart in height than a full grade plus a full step can be
    # either, so this prunes the candidate pairs before any ray is cast.
    reach = grade_cap + step_cap

    # Ground may fall away this far beside the robot before a wheel is in the
    # air.  Tight, and much tighter than what is allowed *along* the direction
    # of travel, because a drop ahead is a slope the robot descends and a drop
    # beside it is one it topples into.  Driving the length of the stair
    # entrance ramp with the right-hand wheels over its 0.13 m edge is what
    # stopped three robots in four; nothing about that drop is visible in the
    # heights of the cells the route passes through.
    lateral_cap = 0.5 * step_cap

    edges: list[tuple[int, int]] = []
    costs: list[float] = []
    for (i, j), here in lookup.items():
        for di, dj in ((1, 0), (0, 1)):
            there = lookup.get((i + di, j + dj))
            if not there:
                continue
            # `_FOOTPRINT_PROBES` is ordered +x, -x, +y, -y, so the pair to
            # test is the one across the direction of travel.
            across = slice(2, 4) if di else slice(0, 2)
            mid_x = ox + (i + 0.5 * di + 0.5) * cx
            mid_y = oy + (j + 0.5 * dj + 0.5) * cy
            for a in here:
                for b in there:
                    if max(drop[a, across].max(), drop[b, across].max()) > lateral_cap:
                        continue
                    za, zb = float(surface[a]), float(surface[b])
                    rise = abs(za - zb)
                    if rise > reach:
                        continue
                    step = _step_between(model, data, mid_x, mid_y, za, zb, cz)
                    if step > step_cap:
                        continue
                    if rise - step > grade_cap:
                        continue
                    # Difficulty feeds the LMDP's passive dynamics later, and
                    # in the meantime it is what stops the planner cutting
                    # corners.  A step is charged far more per metre than a
                    # grade because that is where a wheeled platform actually
                    # fails: given the choice between mounting the side of a
                    # stair ramp and driving four metres round to its foot,
                    # the four metres are the cheaper mistake to make.
                    edges.append((a, b))
                    costs.append(
                        1.0
                        + 3.0 * max(rise - step, 0.0) / grade_cap
                        + _STEP_PENALTY * step / step_cap
                    )

    if not edges:
        return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float64)
    return (
        np.asarray(edges, dtype=np.int32),
        np.asarray(costs, dtype=np.float64),
    )


def _step_between(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    x: float,
    y: float,
    za: float,
    zb: float,
    cell_z: float,
) -> float:
    """The discontinuous part of the rise from one cell's surface to another.

    Zero for a constant grade, the full rise for a clean ledge, and something
    in between for anything else -- read off a single sample of the surface
    halfway between the two cells, compared against the straight line joining
    them.

    A sample that comes back *below* both surfaces is a notch rather than a
    ledge: the two members do not quite meet, which is what a settled stair
    flight and its landing look like a few centimetres apart.  Both ends are
    standable and adjacent cells are one 0.25 m cell apart, so the void can be
    no wider than that and the chassis bridges it; what is left to negotiate
    is the rise between the two ends, and that is what gets charged.
    """
    high, low = max(za, zb), min(za, zb)
    found = _drop(model, data, x, y, high + 0.5 * cell_z, high - low + cell_z)
    if found is None or found < low:
        return high - low
    return 2.0 * abs(found - 0.5 * (za + zb))


# --- queries -----------------------------------------------------------------


def nearest_node(graph: TraversabilityGraph, point: tuple[float, float, float]) -> int:
    """Index of the standable cell closest to `point`."""
    positions = graph.world_positions()
    target = np.asarray(point, dtype=np.float64)
    # Height matters more than plan distance: the cell directly below your
    # feet on the next storey up is not where you are.
    weights = np.array([1.0, 1.0, 2.0])
    distances = (((positions - target) * weights) ** 2).sum(axis=1)
    return int(np.argmin(distances))


def dijkstra_from(
    graph: TraversabilityGraph, start: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cost and predecessor to every node, from one source.

    Separated from `shortest_path` because a swarm asks the same question
    from the same entry point once per robot.  Re-running a full Dijkstra --
    and rebuilding the adjacency list, which dominates it -- for each of them
    is the difference between planning a swarm in a second and in a minute.
    """
    adjacency = graph.adjacency()
    n = graph.n_nodes
    dist = np.full(n, np.inf)
    prev = np.full(n, -1, dtype=np.int64)
    dist[start] = 0.0

    queue: list[tuple[float, int]] = [(0.0, start)]
    seen = np.zeros(n, dtype=bool)
    while queue:
        d, node = heapq.heappop(queue)
        if seen[node]:
            continue
        seen[node] = True
        for neighbour, cost in adjacency[node]:
            nd = d + cost
            if nd < dist[neighbour]:
                dist[neighbour] = nd
                prev[neighbour] = node
                heapq.heappush(queue, (nd, neighbour))
    return dist, prev


def path_from(dist: np.ndarray, prev: np.ndarray, start: int, goal: int) -> list[int]:
    """Unwind a `dijkstra_from` result into a route.  Empty when unreachable."""
    if not np.isfinite(dist[goal]):
        return []
    path = [goal]
    while path[-1] != start:
        path.append(int(prev[path[-1]]))
    return path[::-1]


def shortest_path(graph: TraversabilityGraph, start: int, goal: int) -> list[int]:
    """Dijkstra over the traversability graph.  Empty when unreachable."""
    dist, prev = dijkstra_from(graph, start)
    return path_from(dist, prev, start, goal)


def storeys_reached(
    graph: TraversabilityGraph, path: list[int], storey_height_m: float
) -> set[int]:
    """Which storeys a path visits, by the height of its cells."""
    if not path:
        return set()
    z = graph.surface_z[np.asarray(path, dtype=np.int64)]
    return {int(v) for v in np.floor(z / storey_height_m + 0.25)}


def reachable_storeys(
    graph: TraversabilityGraph, start: int, storey_height_m: float
) -> set[int]:
    """Every storey reachable from `start`, ignoring path cost."""
    adjacency = graph.adjacency()
    seen = np.zeros(graph.n_nodes, dtype=bool)
    seen[start] = True
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbour, _ in adjacency[node]:
            if not seen[neighbour]:
                seen[neighbour] = True
                stack.append(neighbour)
    z = graph.surface_z[seen]
    return {int(v) for v in np.floor(z / storey_height_m + 0.25)}
