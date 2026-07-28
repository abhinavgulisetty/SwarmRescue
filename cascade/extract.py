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
) -> VoxelGrid:
    """Voxelize every colliding box geom at its current pose.

    Boxes are rasterized as oriented boxes, not as their axis-aligned bounds:
    after a collapse debris is rotated, and an AABB of a tilted slab would
    seal voids that the robot can actually drive through -- exactly the
    features that make a rubble field interesting.
    """
    width, depth = footprint_m
    cell = (params.voxel_size_m, params.voxel_size_m, params.voxel_height_m)
    # The grid starts one level *below* z = 0 so that the ground itself can be
    # represented as a solid layer.  Without it there is no floor at storey 0:
    # the ground is a plane geom, this rasterizer only handles boxes, and the
    # lowest standable surface in the whole building came out at z = 3.0 --
    # the top of the first slab.
    origin = (-GRID_MARGIN_M, -GRID_MARGIN_M, -cell[2])
    shape = (
        int(math.ceil((width + 2 * GRID_MARGIN_M) / cell[0])),
        int(math.ceil((depth + 2 * GRID_MARGIN_M) / cell[1])),
        int(math.ceil((total_height_m + GRID_MARGIN_M) / cell[2])) + 1,
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
        _mark_box(
            blocked, origin, cell, center, rot, half, pad, horizontal_only=True
        )

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

    start = [
        max(0, int(math.floor((lo[d] - origin[d]) / cell[d])))
        for d in range(3)
    ]
    stop = [
        min(occupied.shape[d], int(math.ceil((hi[d] - origin[d]) / cell[d])) + 1)
        for d in range(3)
    ]
    if any(stop[d] <= start[d] for d in range(3)):
        return

    axes = [
        origin[d] + (np.arange(start[d], stop[d]) + 0.5) * cell[d]
        for d in range(3)
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
    grid: VoxelGrid, params: TraversabilityParams
) -> TraversabilityGraph:
    """Reduce occupancy to standable cells and the moves between them."""
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

    edges, costs = _connect(cells, surface, grid, params)
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
    nx, ny, nz = grid.shape
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


def _connect(
    cells: np.ndarray,
    surface: np.ndarray,
    grid: VoxelGrid,
    params: TraversabilityParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Join horizontally adjacent standable cells the robot can step between."""
    lookup: dict[tuple[int, int], list[int]] = {}
    for node, (i, j, _) in enumerate(cells):
        lookup.setdefault((int(i), int(j)), []).append(node)

    horizontal = grid.cell[0]
    max_slope = math.tan(math.radians(params.max_slope_deg))
    # Over one cell of run, a traversable grade rises `run * tan(max_slope)`.
    # The step-height cap is about discontinuities *within* a cell, which this
    # resolution cannot see, so it must not also veto a graded slope: a 32
    # degree stair ramp rises 0.25 m per 0.4 m cell, and testing that against
    # the 0.18 m step limit disconnected every staircase in the building and
    # left the whole swarm stuck on the ground floor.
    # Rasterizing a slope onto discrete levels quantizes it: a true 32 degree
    # ramp rises 0.25 m per 0.4 m cell but lands on the 0.1 m lattice as an
    # alternating 0.2 / 0.3 m staircase.  Without an allowance for that, the
    # 0.3 m rises are rejected and every staircase is disconnected.  Half a
    # level is the tolerance; the cost is that the test admits slopes up to
    # ~40 degrees in the worst case, which is the price of the resolution.
    quantization = 0.5 * grid.cell[2]
    max_rise = max(params.max_step_height_m, horizontal * max_slope) + quantization

    edges: list[tuple[int, int]] = []
    costs: list[float] = []
    for (i, j), here in lookup.items():
        for di, dj in ((1, 0), (0, 1)):
            there = lookup.get((i + di, j + dj))
            if not there:
                continue
            for a in here:
                for b in there:
                    rise = abs(float(surface[a] - surface[b]))
                    if rise > max_rise:
                        continue
                    # Difficulty feeds the LMDP's passive dynamics later: a
                    # steep step is passable but should not look free.
                    edges.append((a, b))
                    costs.append(1.0 + 3.0 * rise / max(max_rise, 1e-9))

    if not edges:
        return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float64)
    return (
        np.asarray(edges, dtype=np.int32),
        np.asarray(costs, dtype=np.float64),
    )


# --- queries -----------------------------------------------------------------


def nearest_node(
    graph: TraversabilityGraph, point: tuple[float, float, float]
) -> int:
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


def path_from(
    dist: np.ndarray, prev: np.ndarray, start: int, goal: int
) -> list[int]:
    """Unwind a `dijkstra_from` result into a route.  Empty when unreachable."""
    if not np.isfinite(dist[goal]):
        return []
    path = [goal]
    while path[-1] != start:
        path.append(int(prev[path[-1]]))
    return path[::-1]


def shortest_path(
    graph: TraversabilityGraph, start: int, goal: int
) -> list[int]:
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
