"""Stage 1 -- the floor-plan grammar.

Structure of the generated building, and why it is built this way:

* A single **bay partition** is shared by every storey.  Bays are the
  structural grid, so columns line up vertically from ground to roof.  A
  building whose columns wander between storeys would collapse in a way that
  says more about the generator than about structures.
* Each storey subdivides its bays independently into **rooms** by non-load-
  bearing partitions, so storeys differ without breaking the frame.
* **Vertical connectors** (stairwell, shaft, duct) claim one *cell* of a bay,
  and that cell is chosen storey-independently so the connector is a
  continuous shaft from ground to roof -- the property SS4.3(b) depends on for
  the stack effect.  Connectors are sized to their function; a stairwell is
  ~4.5x3 m, not a whole structural bay.
* A **corridor spine** is carved by a 2-approximate Steiner tree (the minimum
  spanning tree of the metric closure over bay centroids).  Corridors are
  voids cut through partitions, not extra rooms, so rooms stay rectangular
  while the connectivity is organic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import EPS_M, Axis, Rect, SharedWall, manhattan, shared_wall
from .params import BuildingParams, RoomKind
from .rng import stage_rng, substream

#: Width of a carved corridor leg, metres.
CORRIDOR_WIDTH_M: float = 1.8
#: A room this fraction covered by the spine is a corridor, not an office.
_CORRIDOR_COVERAGE_FRACTION: float = 0.45
#: Doors onto a vertical connector are near-certain: stairs need a way in.
_CONNECTOR_DOOR_PROB: float = 0.85


@dataclass(frozen=True, slots=True)
class Room:
    rect: Rect
    kind: RoomKind
    storey: int
    bay: int

    @property
    def is_connector(self) -> bool:
        return self.kind.is_vertical_connector


@dataclass(frozen=True, slots=True)
class ConnectorPlan:
    """A vertical connector's fixed footprint, identical on every storey."""

    bay: int
    kind: RoomKind
    #: The bay's storey-independent subdivision, reused by all storeys.
    cells: tuple[Rect, ...]
    cell_index: int

    @property
    def rect(self) -> Rect:
        return self.cells[self.cell_index]


@dataclass(frozen=True, slots=True)
class Door:
    """An opening cut in the partition between two rooms on one storey."""

    storey: int
    room_a: int
    room_b: int
    wall: SharedWall
    center: tuple[float, float]
    width: float


@dataclass(frozen=True, slots=True)
class Storey:
    index: int
    rooms: tuple[Room, ...]
    doors: tuple[Door, ...]
    corridors: tuple[Rect, ...]

    def rooms_of_kind(self, kind: RoomKind) -> tuple[Room, ...]:
        return tuple(r for r in self.rooms if r.kind is kind)


@dataclass(frozen=True, slots=True)
class Exit:
    """A ground-floor opening through the building envelope.

    These are the egress terminal states of the routing problem: the cells
    where the desirability field is pinned to 1.  A building without them has
    nowhere to lead a survivor.
    """

    #: Normal axis of the envelope wall this pierces.
    axis: Axis
    #: Constant coordinate of that wall.
    at: float
    center: tuple[float, float]
    width: float


@dataclass(frozen=True, slots=True)
class FloorPlan:
    params: BuildingParams
    bays: tuple[Rect, ...]
    storeys: tuple[Storey, ...]
    connectors: tuple[ConnectorPlan, ...]
    exits: tuple[Exit, ...] = ()

    @property
    def footprint(self) -> Rect:
        w, h = self.params.footprint_m
        return Rect(0.0, 0.0, w, h)

    def stairwells(self) -> tuple[ConnectorPlan, ...]:
        return tuple(c for c in self.connectors if c.kind is RoomKind.STAIRWELL)


# --- Stage 1a: BSP -----------------------------------------------------------


def _bsp(
    rect: Rect,
    params: BuildingParams,
    rng: np.random.Generator,
    depth: int,
    min_area: float,
) -> tuple[Rect, ...]:
    """Split `rect` recursively, refusing splits that violate the constraints."""
    if depth <= 0 or rect.area < 2.0 * min_area:
        return (rect,)

    # Split the long axis, so cells tend toward squareness.
    axis = Axis.X if rect.width >= rect.height else Axis.Y
    candidate = _sample_split(rect, axis, params, rng)
    if candidate is None:
        # Long axis refused; try the short one before giving up on this cell.
        axis = axis.other
        candidate = _sample_split(rect, axis, params, rng)
    if candidate is None:
        return (rect,)

    lo, hi = rect.split(axis, candidate)
    if min(lo.area, hi.area) < min_area:
        return (rect,)

    return _bsp(lo, params, rng, depth - 1, min_area) + _bsp(
        hi, params, rng, depth - 1, min_area
    )


def _sample_split(
    rect: Rect, axis: Axis, params: BuildingParams, rng: np.random.Generator
) -> float | None:
    """Sample a split coordinate honouring side, area and aspect constraints.

    Returns None when no admissible split exists on this axis.
    """
    extent = rect.extent(axis)
    if extent < 2.0 * params.min_room_side_m:
        return None

    origin = rect.x0 if axis is Axis.X else rect.y0
    jitter = params.split_jitter
    frac_lo = max(params.min_room_side_m / extent, 0.5 - jitter)
    frac_hi = min(1.0 - params.min_room_side_m / extent, 0.5 + jitter)
    if frac_lo >= frac_hi:
        return None

    at = origin + extent * float(rng.uniform(frac_lo, frac_hi))
    lo, hi = rect.split(axis, at)
    if max(lo.aspect, hi.aspect) > params.max_room_aspect:
        return None
    return at


def partition_bays(
    footprint: Rect, params: BuildingParams, rng: np.random.Generator
) -> tuple[Rect, ...]:
    """Recursive BSP down to bay granularity (the shared structural grid)."""
    return _bsp(footprint, params, rng, params.bay_depth, params.min_bay_area_m2)


def subdivide_bay(
    bay: Rect, params: BuildingParams, rng: np.random.Generator
) -> tuple[Rect, ...]:
    """Subdivide one bay into rooms by non-load-bearing partitions."""
    return _bsp(bay, params, rng, params.room_depth, params.min_room_area_m2)


# --- Stage 1b: vertical connectors -------------------------------------------


def plan_connectors(
    bays: tuple[Rect, ...], params: BuildingParams, rng: np.random.Generator
) -> tuple[ConnectorPlan, ...]:
    """Choose bays and cells for the vertical connectors, once for all storeys.

    Stairwells are biased toward the building perimeter, the way real egress
    stairs are; shafts and ducts take whatever bays remain.  The chosen bay's
    subdivision is frozen here and replayed on every storey, which is what
    makes the connector a continuous shaft rather than a stack of holes that
    happen to overlap.
    """
    wanted: list[RoomKind] = (
        [RoomKind.STAIRWELL] * params.n_stairwells
        + [RoomKind.SHAFT] * params.n_shafts
        + [RoomKind.DUCT] * params.n_ducts
    )
    # Never consume every bay: a building of nothing but shafts has no rooms.
    wanted = wanted[: max(0, len(bays) - 1)]
    if not wanted:
        return ()

    hull = _hull(bays)
    remaining = list(range(len(bays)))
    scores = np.array([_perimeter_affinity(bays[i], hull) for i in remaining])

    plans: list[ConnectorPlan] = []
    for kind in wanted:
        if not remaining:
            break
        if kind is RoomKind.STAIRWELL:
            weights = scores - scores.min() + 1e-6
            pick = int(rng.choice(len(remaining), p=weights / weights.sum()))
        else:
            pick = int(rng.integers(len(remaining)))
        bay_index = remaining[pick]

        cells = subdivide_bay(bays[bay_index], params, rng)
        host, width, depth = _choose_host_cell(cells, kind, params)
        shaft, remainder = _carve_connector(cells[host], width, depth, rng)
        # Guillotine the host cell into the shaft plus rectangular leftovers,
        # so a duct is duct-sized instead of inheriting a whole room.
        final_cells = (*cells[:host], shaft, *remainder, *cells[host + 1 :])
        plans.append(
            ConnectorPlan(bay=bay_index, kind=kind, cells=final_cells, cell_index=host)
        )
        remaining.pop(pick)
        scores = np.delete(scores, pick)

    return tuple(sorted(plans, key=lambda p: p.bay))


def _connector_size(kind: RoomKind, params: BuildingParams) -> tuple[float, float]:
    sizes = {
        RoomKind.STAIRWELL: params.stairwell_size_m,
        RoomKind.SHAFT: params.shaft_size_m,
        RoomKind.DUCT: params.duct_size_m,
    }
    if kind not in sizes:
        raise ValueError(f"{kind} is not a vertical connector kind")
    return sizes[kind]


def _choose_host_cell(
    cells: tuple[Rect, ...], kind: RoomKind, params: BuildingParams
) -> tuple[int, float, float]:
    """Pick the cell to cut the connector from, and the footprint to cut.

    Prefers a cell that fits the connector at full size in either orientation,
    so a stairwell is not silently clamped down by a narrow host.  Falls back
    to the roomiest cell with a clamped footprint when nothing fits.
    """
    want_w, want_d = _connector_size(kind, params)
    tol = 1e-9

    fitting = [
        (abs(cell.area - want_w * want_d), index, w, d)
        for index, cell in enumerate(cells)
        for w, d in ((want_w, want_d), (want_d, want_w))
        if cell.width >= w - tol and cell.height >= d - tol
    ]
    if fitting:
        _, index, w, d = min(fitting)
        return index, w, d

    roomiest = int(np.argmax([c.area for c in cells]))
    cell = cells[roomiest]
    return roomiest, min(want_w, cell.width), min(want_d, cell.height)


def _carve_connector(
    cell: Rect, width: float, depth: float, rng: np.random.Generator
) -> tuple[Rect, tuple[Rect, ...]]:
    """Cut a `width` x `depth` connector from a corner of `cell`.

    Returns the connector rectangle and the rectangular remainder.  A corner
    cut admits an exact guillotine decomposition into at most two leftover
    rectangles, so every room in the plan stays a rectangle -- which the
    structural and voxel stages both rely on.
    """
    w = min(width, cell.width)
    d = min(depth, cell.height)

    on_right = bool(rng.integers(2))
    on_top = bool(rng.integers(2))
    x0 = cell.x1 - w if on_right else cell.x0
    y0 = cell.y1 - d if on_top else cell.y0
    shaft = Rect(x0, y0, x0 + w, y0 + d)

    remainder: list[Rect] = []
    # Vertical strip beside the shaft, spanning the cell's full height.
    if on_right:
        if cell.x1 - w - cell.x0 > 0.0:
            remainder.append(Rect(cell.x0, cell.y0, cell.x1 - w, cell.y1))
    elif cell.x1 - (cell.x0 + w) > 0.0:
        remainder.append(Rect(cell.x0 + w, cell.y0, cell.x1, cell.y1))

    # Horizontal strip above/below the shaft, spanning only the shaft's width.
    if on_top:
        if cell.y1 - d - cell.y0 > 0.0:
            remainder.append(Rect(x0, cell.y0, x0 + w, cell.y1 - d))
    elif cell.y1 - (cell.y0 + d) > 0.0:
        remainder.append(Rect(x0, cell.y0 + d, x0 + w, cell.y1))

    return shaft, tuple(remainder)


def _hull(bays: tuple[Rect, ...]) -> Rect:
    return Rect(
        min(b.x0 for b in bays),
        min(b.y0 for b in bays),
        max(b.x1 for b in bays),
        max(b.y1 for b in bays),
    )


def _perimeter_affinity(bay: Rect, hull: Rect) -> float:
    """How close a bay sits to the building envelope, in [0, 1]."""
    cx, cy = bay.center
    dx = min(cx - hull.x0, hull.x1 - cx) / max(hull.width, 1e-9)
    dy = min(cy - hull.y0, hull.y1 - cy) / max(hull.height, 1e-9)
    return 1.0 - 2.0 * min(dx, dy)


# --- Stage 1c: the corridor spine (2-approximate Steiner tree) ---------------


def carve_corridor_spine(bays: tuple[Rect, ...]) -> tuple[Rect, ...]:
    """Carve L-shaped corridor legs along the MST of the metric closure.

    Terminals are bay centroids.  The MST of the complete graph under the
    Manhattan metric is the textbook 2-approximation to the Steiner tree, and
    Manhattan legs are exactly the axis-aligned corridors a building has.
    """
    terminals = [b.center for b in bays]
    if len(terminals) < 2:
        return ()

    half = 0.5 * CORRIDOR_WIDTH_M
    legs: list[Rect] = []
    for i, j in _mst_edges(terminals):
        (ax, ay), (bx, by) = terminals[i], terminals[j]
        # Horizontal leg then vertical leg -- an L through the corner (bx, ay).
        legs.append(Rect(min(ax, bx) - half, ay - half, max(ax, bx) + half, ay + half))
        legs.append(Rect(bx - half, min(ay, by) - half, bx + half, max(ay, by) + half))
    return tuple(legs)


def _mst_edges(points: list[tuple[float, float]]) -> tuple[tuple[int, int], ...]:
    """Prim's algorithm under the Manhattan metric.

    Implemented directly rather than via networkx/scipy: the terminal count is
    small, and a dependency-free routine keeps this deterministic and cheap.
    """
    n = len(points)
    if n < 2:
        return ()

    in_tree = [False] * n
    best_cost = [float("inf")] * n
    best_from = [-1] * n
    in_tree[0] = True
    for k in range(1, n):
        best_cost[k] = manhattan(points[0], points[k])
        best_from[k] = 0

    edges: list[tuple[int, int]] = []
    for _ in range(n - 1):
        pending = [k for k in range(n) if not in_tree[k]]
        if not pending:
            break
        nxt = min(pending, key=lambda k: best_cost[k])
        in_tree[nxt] = True
        edges.append((min(best_from[nxt], nxt), max(best_from[nxt], nxt)))
        for k in range(n):
            if in_tree[k]:
                continue
            cost = manhattan(points[nxt], points[k])
            if cost < best_cost[k]:
                best_cost[k] = cost
                best_from[k] = nxt
    return tuple(edges)


# --- Stage 1d: assembling a storey -------------------------------------------


def build_storey(
    storey: int,
    bays: tuple[Rect, ...],
    connectors: tuple[ConnectorPlan, ...],
    corridors: tuple[Rect, ...],
    params: BuildingParams,
    rng: np.random.Generator,
) -> Storey:
    """Subdivide bays into rooms, apply the spine, hang doors."""
    by_bay = {plan.bay: plan for plan in connectors}

    rooms: list[Room] = []
    for bay_index, bay in enumerate(bays):
        plan = by_bay.get(bay_index)
        if plan is not None:
            # Replay the frozen subdivision so the shaft is vertically aligned.
            for cell_index, cell in enumerate(plan.cells):
                kind = plan.kind if cell_index == plan.cell_index else RoomKind.OFFICE
                rooms.append(Room(rect=cell, kind=kind, storey=storey, bay=bay_index))
            continue
        for cell in subdivide_bay(bay, params, rng):
            rooms.append(
                Room(rect=cell, kind=RoomKind.OFFICE, storey=storey, bay=bay_index)
            )

    rooms_t = _retag_corridor_rooms(tuple(rooms), corridors)
    doors = place_doors(storey, rooms_t, corridors, params, rng)
    return Storey(index=storey, rooms=rooms_t, doors=doors, corridors=corridors)


def _retag_corridor_rooms(
    rooms: tuple[Room, ...], corridors: tuple[Rect, ...]
) -> tuple[Room, ...]:
    """Relabel a room the spine dominates as CORRIDOR.

    Fuel load follows the kind, so a hallway must not be charged as an office.
    Rebuilt immutably -- rooms are never edited in place.
    """
    if not corridors:
        return rooms
    out: list[Room] = []
    for room in rooms:
        if room.is_connector:
            out.append(room)
            continue
        covered = sum(_intersection_area(room.rect, c) for c in corridors)
        kind = (
            RoomKind.CORRIDOR
            if covered > _CORRIDOR_COVERAGE_FRACTION * room.rect.area
            else room.kind
        )
        out.append(Room(rect=room.rect, kind=kind, storey=room.storey, bay=room.bay))
    return tuple(out)


def _intersection_area(a: Rect, b: Rect) -> float:
    w = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    h = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    return w * h


# --- Stage 1e: doors ---------------------------------------------------------


def place_doors(
    storey: int,
    rooms: tuple[Room, ...],
    corridors: tuple[Rect, ...],
    params: BuildingParams,
    rng: np.random.Generator,
) -> tuple[Door, ...]:
    """Hang doors on shared walls, then repair connectivity.

    Probability decays with wall length, so a long shared wall is not riddled
    with doorways.  Afterwards the room graph induced by doors *and* corridors
    is forced connected -- an unreachable room is a generator bug, not an
    interesting scenario.
    """
    candidates: list[tuple[int, int, SharedWall]] = []
    for a in range(len(rooms)):
        for b in range(a + 1, len(rooms)):
            wall = shared_wall(rooms[a].rect, rooms[b].rect, params.door_width_m)
            if wall is not None:
                candidates.append((a, b, wall))

    chosen: list[tuple[int, int, SharedWall]] = []
    for a, b, wall in candidates:
        prob = float(np.exp(-wall.length / params.door_prob_length_scale_m))
        # Connectors are entered deliberately, not incidentally.
        if rooms[a].is_connector or rooms[b].is_connector:
            prob = max(prob, _CONNECTOR_DOOR_PROB)
        if rng.random() < prob:
            chosen.append((a, b, wall))

    chosen = _repair_connectivity(chosen, candidates, rooms, corridors)
    return tuple(_make_door(storey, a, b, wall, params, rng) for a, b, wall in chosen)


def _make_door(
    storey: int,
    a: int,
    b: int,
    wall: SharedWall,
    params: BuildingParams,
    rng: np.random.Generator,
) -> Door:
    half = 0.5 * params.door_width_m
    lo, hi = wall.lo + half, wall.hi - half
    along = float(rng.uniform(lo, hi)) if hi > lo else 0.5 * (wall.lo + wall.hi)
    return Door(
        storey=storey,
        room_a=a,
        room_b=b,
        wall=wall,
        center=wall.point_at(along),
        width=params.door_width_m,
    )


def _repair_connectivity(
    chosen: list[tuple[int, int, SharedWall]],
    candidates: list[tuple[int, int, SharedWall]],
    rooms: tuple[Room, ...],
    corridors: tuple[Rect, ...],
) -> list[tuple[int, int, SharedWall]]:
    """Add the fewest extra doors needed to make every room reachable."""
    parent = list(range(len(rooms)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        parent[ry] = rx
        return True

    for a, b, _ in chosen:
        union(a, b)

    # Rooms sharing a corridor leg are connected without needing a door.
    for leg in corridors:
        touched = [i for i, r in enumerate(rooms) if r.rect.overlaps(leg)]
        for other in touched[1:]:
            union(touched[0], other)

    result = list(chosen)
    for a, b, wall in candidates:
        if find(a) != find(b):
            union(a, b)
            result.append((a, b, wall))
    return result


# --- Entry point -------------------------------------------------------------


def build_floorplan(seed: int, params: BuildingParams) -> FloorPlan:
    """Generate the complete multi-storey floor plan for `seed`."""
    rng = stage_rng(seed, "floorplan")
    w, h = params.footprint_m
    footprint = Rect(0.0, 0.0, w, h)

    bays = partition_bays(footprint, params, rng)
    connectors = plan_connectors(bays, params, rng)
    corridors = carve_corridor_spine(bays)

    storeys = tuple(
        build_storey(k, bays, connectors, corridors, params, substream(rng, k))
        for k in range(params.n_storeys)
    )
    exits = place_exits(storeys[0], footprint, params, rng)
    return FloorPlan(
        params=params,
        bays=bays,
        storeys=storeys,
        connectors=connectors,
        exits=exits,
    )


def place_exits(
    ground: Storey,
    footprint: Rect,
    params: BuildingParams,
    rng: np.random.Generator,
) -> tuple[Exit, ...]:
    """Punch exits through the ground-floor envelope.

    Candidates are ground-floor room edges that lie on the building boundary
    and are long enough to hold an opening.  Exits are spread across different
    facades where possible: two doors side by side on one wall give the swarm
    a single egress direction and make the routing problem degenerate.
    """
    candidates: list[tuple[Axis, float, float, float]] = []
    for room in ground.rooms:
        if room.is_connector:
            continue
        r = room.rect
        for axis, at, lo, hi, boundary in (
            (Axis.Y, r.y0, r.x0, r.x1, footprint.y0),
            (Axis.Y, r.y1, r.x0, r.x1, footprint.y1),
            (Axis.X, r.x0, r.y0, r.y1, footprint.x0),
            (Axis.X, r.x1, r.y0, r.y1, footprint.x1),
        ):
            on_envelope = abs(at - boundary) <= EPS_M
            if on_envelope and (hi - lo) >= params.exit_width_m + 2 * EPS_M:
                candidates.append((axis, at, lo, hi))

    if not candidates:
        return ()

    # Group by facade so exits can be spread across the building.
    by_facade: dict[tuple[str, float], list[tuple[Axis, float, float, float]]] = {}
    for axis, at, lo, hi in candidates:
        by_facade.setdefault((axis.value, at), []).append((axis, at, lo, hi))

    facades = sorted(by_facade)
    order = rng.permutation(len(facades))
    exits: list[Exit] = []
    for index in order:
        if len(exits) >= params.n_exits:
            break
        segments = by_facade[facades[int(index)]]
        axis, at, lo, hi = segments[int(rng.integers(len(segments)))]
        half = 0.5 * params.exit_width_m
        along = float(rng.uniform(lo + half, hi - half))
        center = (at, along) if axis is Axis.X else (along, at)
        exits.append(Exit(axis=axis, at=at, center=center, width=params.exit_width_m))
    return tuple(exits)
