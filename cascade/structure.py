"""Stage 2 -- the structural graph.

Columns, beams and slabs are instantiated as rigid bodies joined by weld
constraints, forming a load-bearing graph.  The load path is explicit:

    ground --weld--> column(storey 0) --weld--> slab(level 1)
                          ^                          |
                          +-- column(storey 1) <-----+   ... and so on

Releasing a weld therefore removes real support, and MuJoCo does the rest.
Nothing here scripts a collapse; it only builds something that *can* collapse.

Each member carries a mass, a yield threshold and a thermal mass, because the
progressive-degradation model of SS4.3(c) integrates thermal damage per member
and releases its weld when the damage crosses that member's threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .floorplan import FloorPlan, Room
from .geometry import EPS_M, Axis, Rect
from .params import (
    CONCRETE_DENSITY_KG_M3,
    PARTITION_DENSITY_KG_M3,
    BuildingParams,
)

#: Specific heat capacity, J/(kg K).  Used for the thermal-damage integral.
CONCRETE_SPECIFIC_HEAT_J_KG_K: float = 880.0
PARTITION_SPECIFIC_HEAT_J_KG_K: float = 1200.0

#: Cold compressive strength of the concrete members, Pa (a C30/37-class mix,
#: derated for an ageing structure).
CONCRETE_YIELD_STRESS_PA: float = 18.0e6
PARTITION_YIELD_STRESS_PA: float = 1.5e6

#: Residual strength fraction of the members at the moment of the collapse.
#:
#: These are members in a burning, already-damaged building, not fresh ones.
#: EN 1992-1-2 gives a concrete strength reduction factor k_c(theta) of about
#: 0.45 at 600 C, and heated members also lose section to spalling; 0.40 sits
#: just inside that.
#:
#: This is calibrated against measurement, not guessed.  On the reference
#: model the heaviest-loaded ground column carries 575 kN intact, and 811 kN
#: after a blast destroys five of the eighteen ground columns and their load
#: redistributes.  The factor has to sit between those two numbers: above 811
#: kN capacity the frame absorbs the blast and nothing falls, below 575 kN it
#: collapses under its own weight with no blast at all.
#:
#:     factor 0.40 -> 882 kN : survives the blast (ratio 0.92, no cascade)
#:     factor 0.35 -> 772 kN : stands intact (margin 1.34x), cascades on blast
#:
#: Settled at 0.45 (993 kN) once connection failure was modelled.  Members
#: this strong let the *undamaged* part of the frame carry the redistributed
#: load and stay up, while the section that lost its columns separates at its
#: connections and falls -- which is what makes the collapse partial.  When
#: members were the only thing that could fail, the same value meant nothing
#: fell at all.
#:
#: 0.35 also sits sensibly against EN 1992-1-2, whose concrete strength
#: reduction factor k_c(theta) is about 0.45 at 600 C before section loss to
#: spalling is accounted for.
FIRE_DAMAGED_STRENGTH_FACTOR: float = 0.45

#: Coordinates within this distance are the same structural node, metres.
NODE_MERGE_TOL_M: float = 1e-6

#: A column standing within this distance of a bay is taken to bear on it.
_SUPPORT_TOL_M: float = 1e-6

#: Deflection head joint: the gap left between a partition's head and the slab
#: soffit above it, metres.
#:
#: Real construction detrails partitions this way precisely so they do NOT
#: pick up structural load as the slab deflects.  Modelling them floor-to-
#: soffit instead made every partition a load-bearing column: ~26 walls per
#: storey silently propped the frame up, and eleven of eighteen ground columns
#: could be destroyed without the building moving at all.
WALL_HEAD_GAP_M: float = 0.025

#: Walls are not built across a corridor leg or a doorway; this pads the gap
#: so a robot is not scraped by a wall edge exactly at the opening, metres.
OPENING_CLEARANCE_M: float = 0.05


class MemberKind(str, Enum):
    COLUMN = "column"
    BEAM = "beam"
    SLAB = "slab"
    WALL = "wall"

    @property
    def is_load_bearing(self) -> bool:
        return self in (MemberKind.COLUMN, MemberKind.BEAM, MemberKind.SLAB)


@dataclass(frozen=True, slots=True)
class Member:
    """One rigid structural element, as an axis-aligned box."""

    name: str
    kind: MemberKind
    #: Centre of the box in world coordinates, metres.
    center: tuple[float, float, float]
    #: Full extents (not half-extents) in metres.
    size: tuple[float, float, float]
    storey: int
    density: float

    @property
    def half_extents(self) -> tuple[float, float, float]:
        return (0.5 * self.size[0], 0.5 * self.size[1], 0.5 * self.size[2])

    @property
    def volume_m3(self) -> float:
        return self.size[0] * self.size[1] * self.size[2]

    @property
    def mass_kg(self) -> float:
        return self.density * self.volume_m3

    @property
    def specific_heat_j_kg_k(self) -> float:
        return (
            CONCRETE_SPECIFIC_HEAT_J_KG_K
            if self.density >= CONCRETE_DENSITY_KG_M3 - 1.0
            else PARTITION_SPECIFIC_HEAT_J_KG_K
        )

    @property
    def thermal_mass_j_per_k(self) -> float:
        """Energy needed to raise this member by one kelvin."""
        return self.mass_kg * self.specific_heat_j_kg_k

    @property
    def yield_threshold_n(self) -> float:
        """Axial load the member carries before failing, newtons.

        Cross-section is taken normal to the member's slenderest axis, which
        for a column is its footprint and for a slab its thickness plane.
        """
        stress = (
            CONCRETE_YIELD_STRESS_PA
            if self.kind.is_load_bearing
            else PARTITION_YIELD_STRESS_PA
        )
        sx, sy, sz = self.size
        cross_section = min(sx * sy, sx * sz, sy * sz)
        return FIRE_DAMAGED_STRENGTH_FACTOR * stress * cross_section


@dataclass(frozen=True, slots=True)
class Weld:
    """An equality constraint joining two members (or a member to the world).

    `body_b is None` means the member is anchored to the world -- the ground
    fixing of a storey-0 column.  Releasing a weld at runtime via
    `data.eq_active` is the primitive the whole collapse model rests on.
    """

    name: str
    body_a: str
    body_b: str | None


@dataclass(frozen=True, slots=True)
class Structure:
    params: BuildingParams
    members: tuple[Member, ...]
    welds: tuple[Weld, ...]

    def of_kind(self, kind: MemberKind) -> tuple[Member, ...]:
        return tuple(m for m in self.members if m.kind is kind)

    @property
    def total_mass_kg(self) -> float:
        return sum(m.mass_kg for m in self.members)

    def member_index(self) -> dict[str, Member]:
        return {m.name: m for m in self.members}


# --- structural node extraction ----------------------------------------------


def column_nodes(bays: tuple[Rect, ...]) -> tuple[tuple[float, float], ...]:
    """Deduplicated bay corners -- one column per structural node.

    Four bays meeting at a point share a single column, which is why this
    deduplicates rather than emitting four coincident bodies.
    """
    seen: dict[tuple[int, int], tuple[float, float]] = {}
    for bay in bays:
        for x, y in (
            (bay.x0, bay.y0),
            (bay.x1, bay.y0),
            (bay.x0, bay.y1),
            (bay.x1, bay.y1),
        ):
            seen.setdefault(_quantize(x, y), (x, y))
    return tuple(seen[k] for k in sorted(seen))


def beam_spans(
    bays: tuple[Rect, ...],
) -> tuple[tuple[Axis, float, float, float], ...]:
    """Deduplicated bay edges as (axis, constant coord, lo, hi) spans.

    `axis` is the direction the beam *runs* along.
    """
    seen: dict[tuple[int, int, int], tuple[Axis, float, float, float]] = {}
    for bay in bays:
        edges = (
            (Axis.X, bay.y0, bay.x0, bay.x1),
            (Axis.X, bay.y1, bay.x0, bay.x1),
            (Axis.Y, bay.x0, bay.y0, bay.y1),
            (Axis.Y, bay.x1, bay.y0, bay.y1),
        )
        for axis, at, lo, hi in edges:
            key = (0 if axis is Axis.X else 1, _q(at), _q(lo) ^ _q(hi))
            seen.setdefault(key, (axis, at, lo, hi))
    return tuple(seen[k] for k in sorted(seen))


def _quantize(x: float, y: float) -> tuple[int, int]:
    return (_q(x), _q(y))


def _q(v: float) -> int:
    return int(round(v / NODE_MERGE_TOL_M))


# --- member construction -----------------------------------------------------


def build_structure(plan: FloorPlan) -> Structure:
    """Instantiate every structural member and the welds that hold them up."""
    params = plan.params
    members: list[Member] = []
    welds: list[Weld] = []

    nodes = column_nodes(plan.bays)
    spans = beam_spans(plan.bays)
    h = params.storey_height_m

    for storey in range(params.n_storeys):
        z_floor = storey * h
        z_ceiling = z_floor + h

        members.extend(_columns(nodes, storey, z_floor, params))
        members.extend(_beams(spans, storey, z_ceiling, params))
        members.extend(_slabs(plan.bays, storey, z_ceiling, params))

        walls = _walls(
            plan.storeys[storey].rooms,
            plan.storeys[storey].corridors,
            plan,
            storey,
            z_floor,
            params,
        )
        members.extend(walls)
        # Partitions rest on the floor beneath them: the ground on storey 0,
        # otherwise the slab that forms this storey's floor.
        welds.extend(_wall_welds(walls, storey, plan))

    welds.extend(_load_path_welds(nodes, spans, plan, params))
    return Structure(params=params, members=tuple(members), welds=tuple(welds))


def _wall_welds(
    walls: list[Member], storey: int, plan: FloorPlan
) -> list[Weld]:
    """Anchor each partition to the floor it stands on."""
    welds: list[Weld] = []
    for wall in walls:
        x, y, _ = wall.center
        below = None if storey == 0 else _slab_under(x, y, storey - 1, plan)
        welds.append(Weld(name=f"w_{wall.name}", body_a=wall.name, body_b=below))
    return welds


def _columns(
    nodes: tuple[tuple[float, float], ...],
    storey: int,
    z_floor: float,
    params: BuildingParams,
) -> list[Member]:
    # A column rises from its floor to the underside of the slab it carries,
    # not through it -- overlapping the slab would spawn an interpenetrating
    # contact pair at t = 0 and fight the weld holding them together.
    clear_height = params.storey_height_m - params.slab_thickness_m
    side = params.column_side_m
    return [
        Member(
            name=f"col_s{storey}_n{i}",
            kind=MemberKind.COLUMN,
            center=(x, y, z_floor + 0.5 * clear_height),
            size=(side, side, clear_height),
            storey=storey,
            density=CONCRETE_DENSITY_KG_M3,
        )
        for i, (x, y) in enumerate(nodes)
    ]


def _beams(
    spans: tuple[tuple[Axis, float, float, float], ...],
    storey: int,
    z_ceiling: float,
    params: BuildingParams,
) -> list[Member]:
    w, d = params.beam_width_m, params.beam_depth_m
    out: list[Member] = []
    for i, (axis, at, lo, hi) in enumerate(spans):
        length = hi - lo
        mid = 0.5 * (lo + hi)
        # The beam sits directly beneath the slab it carries.
        cz = z_ceiling - params.slab_thickness_m - 0.5 * d
        if axis is Axis.X:
            center, size = (mid, at, cz), (length, w, d)
        else:
            center, size = (at, mid, cz), (w, length, d)
        out.append(
            Member(
                name=f"beam_s{storey}_e{i}",
                kind=MemberKind.BEAM,
                center=center,
                size=size,
                storey=storey,
                density=CONCRETE_DENSITY_KG_M3,
            )
        )
    return out


def _slabs(
    bays: tuple[Rect, ...], storey: int, z_ceiling: float, params: BuildingParams
) -> list[Member]:
    """The slab above `storey` -- i.e. the floor of the storey above, or the
    roof for the topmost storey."""
    t = params.slab_thickness_m
    out: list[Member] = []
    for i, bay in enumerate(bays):
        cx, cy = bay.center
        out.append(
            Member(
                name=f"slab_s{storey}_b{i}",
                kind=MemberKind.SLAB,
                center=(cx, cy, z_ceiling - 0.5 * t),
                size=(bay.width, bay.height, t),
                storey=storey,
                density=CONCRETE_DENSITY_KG_M3,
            )
        )
    return out


def _walls(
    rooms: tuple[Room, ...],
    corridors: tuple[Rect, ...],
    plan: FloorPlan,
    storey: int,
    z_floor: float,
    params: BuildingParams,
) -> list[Member]:
    """Walls on room boundaries and the envelope, broken by openings.

    Only the low-coordinate edges of each room are emitted, so a boundary
    shared by two rooms yields one wall rather than two coincident ones.  The
    exception is the building envelope: a room's high edge is emitted when it
    lies on the footprint boundary, because nothing else will ever emit it.

    Omitting that exception left two of the four facades missing entirely --
    the building was open on the +x and +y sides.  That is not cosmetic: the
    stack effect of SS4.3(b) needs an enclosure to drive smoke up the
    stairwells, and an open-sided building simply vents.  The RSSI model's
    wall-attenuation raycasts had nothing to cross either.
    """
    # Partitions stop short of the soffit (deflection head joint), so they
    # carry no vertical load and cannot substitute for a destroyed column.
    h = params.storey_height_m - params.slab_thickness_m - WALL_HEAD_GAP_M
    t = params.wall_thickness_m
    doors = plan.storeys[storey].doors
    # Exits pierce the envelope on the ground floor only.
    exits = plan.exits if storey == 0 else ()
    openings = _opening_intervals(doors, exits)

    footprint = plan.footprint
    out: list[Member] = []
    for room_index, room in enumerate(rooms):
        r = room.rect
        edges = [
            (Axis.Y, r.y0, r.x0, r.x1),  # wall running along x, at y = y0
            (Axis.X, r.x0, r.y0, r.y1),  # wall running along y, at x = x0
        ]
        # The envelope's far faces have no neighbouring room to emit them.
        if abs(r.y1 - footprint.y1) <= EPS_M:
            edges.append((Axis.Y, r.y1, r.x0, r.x1))
        if abs(r.x1 - footprint.x1) <= EPS_M:
            edges.append((Axis.X, r.x1, r.y0, r.y1))

        for axis, at, lo, hi in edges:
            for seg_lo, seg_hi in _split_by_openings(
                axis, at, lo, hi, openings, corridors
            ):
                if seg_hi - seg_lo < params.door_width_m * 0.5:
                    continue
                length = seg_hi - seg_lo
                mid = 0.5 * (seg_lo + seg_hi)
                if axis is Axis.Y:
                    center, size = (mid, at, z_floor + 0.5 * h), (length, t, h)
                else:
                    center, size = (at, mid, z_floor + 0.5 * h), (t, length, h)
                out.append(
                    Member(
                        name=f"wall_s{storey}_r{room_index}_"
                        f"{axis.value}_{len(out)}",
                        kind=MemberKind.WALL,
                        center=center,
                        size=size,
                        storey=storey,
                        density=PARTITION_DENSITY_KG_M3,
                    )
                )
    return out


def _opening_intervals(
    doors: tuple, exits: tuple = ()
) -> list[tuple[Axis, float, float, float]]:
    """Doorway and exit gaps, as (normal axis, coordinate, lo, hi).

    Corridors are handled separately in `_split_by_openings`: a corridor is a
    2-D region, and flattening it into this per-axis form is what produced a
    coordinate mix-up that punched holes in walls at meaningless positions.
    """
    out: list[tuple[Axis, float, float, float]] = []
    for door_out in exits:
        half = 0.5 * door_out.width + OPENING_CLEARANCE_M
        along = door_out.center[1] if door_out.axis is Axis.X else door_out.center[0]
        out.append((door_out.axis, door_out.at, along - half, along + half))
    for door in doors:
        half = 0.5 * door.width + OPENING_CLEARANCE_M
        along = door.center[1] if door.wall.axis is Axis.X else door.center[0]
        # A door in a wall whose normal is X interrupts walls running along Y.
        out.append((door.wall.axis, door.wall.at, along - half, along + half))
    return out


def _split_by_openings(
    axis: Axis,
    at: float,
    lo: float,
    hi: float,
    openings: list[tuple[Axis, float, float, float]],
    corridors: tuple[Rect, ...] = (),
) -> list[tuple[float, float]]:
    """Subtract every applicable opening from the wall span [lo, hi].

    `axis` is the wall's *normal*: `Axis.Y` means the wall runs along x at the
    constant coordinate `at`, and `[lo, hi]` is its extent in x.
    """
    cuts: list[tuple[float, float]] = []
    for op_axis, op_at, op_lo, op_hi in openings:
        if op_axis is axis and abs(op_at - at) <= EPS_M:
            cuts.append((op_lo, op_hi))

    # A corridor cuts this wall only where it actually crosses it, and the gap
    # it leaves is its extent along the wall -- not across it.
    for leg in corridors:
        if axis is Axis.Y:
            if leg.y0 - EPS_M <= at <= leg.y1 + EPS_M:
                cuts.append((leg.x0, leg.x1))
        elif leg.x0 - EPS_M <= at <= leg.x1 + EPS_M:
            cuts.append((leg.y0, leg.y1))

    segments = [(lo, hi)]
    for cut_lo, cut_hi in cuts:
        nxt: list[tuple[float, float]] = []
        for seg_lo, seg_hi in segments:
            if cut_hi <= seg_lo or cut_lo >= seg_hi:
                nxt.append((seg_lo, seg_hi))
                continue
            if cut_lo > seg_lo:
                nxt.append((seg_lo, cut_lo))
            if cut_hi < seg_hi:
                nxt.append((cut_hi, seg_hi))
        segments = nxt
    return segments


# --- the load path -----------------------------------------------------------


def _load_path_welds(
    nodes: tuple[tuple[float, float], ...],
    spans: tuple[tuple[Axis, float, float, float], ...],
    plan: FloorPlan,
    params: BuildingParams,
) -> list[Weld]:
    """Wire the load path: ground -> columns -> slabs -> columns above.

    Each weld is individually releasable, which is what makes a collapse a
    physical consequence rather than an animation.
    """
    welds: list[Weld] = []
    for storey in range(params.n_storeys):
        for i, (x, y) in enumerate(nodes):
            col = f"col_s{storey}_n{i}"
            if storey == 0:
                # Storey-0 columns are fixed to the ground.
                welds.append(Weld(name=f"w_foot_{col}", body_a=col, body_b=None))
            else:
                for k, slab in enumerate(_slabs_at(x, y, storey - 1, plan)):
                    welds.append(
                        Weld(name=f"w_foot_{col}_{k}", body_a=col, body_b=slab)
                    )
            # The head carries every slab bearing on this node.
            for k, slab in enumerate(_slabs_at(x, y, storey, plan)):
                welds.append(Weld(name=f"w_head_{col}_{k}", body_a=col, body_b=slab))

        for i, (axis, at, lo, hi) in enumerate(spans):
            beam = f"beam_s{storey}_e{i}"
            mid = 0.5 * (lo + hi)
            bx, by = (mid, at) if axis is Axis.X else (at, mid)
            for k, slab in enumerate(_slabs_at(bx, by, storey, plan)):
                welds.append(Weld(name=f"w_{beam}_{k}", body_a=beam, body_b=slab))

    return welds


def overlapping_pairs(
    members: tuple[Member, ...], tol: float = 1e-4
) -> tuple[tuple[str, str], ...]:
    """Member pairs whose boxes interpenetrate in the reference configuration.

    Members that overlap while intact are part of one rigid assembly -- a beam
    framing into a column, a wall meeting the beam above it.  Their contact
    pairs are excluded in the emitted model, because a contact and the weld
    joining the same two bodies would otherwise fight each other from t = 0.

    Honest limitation: an excluded pair stays excluded after the collapse
    separates it, so two members that were adjacent when intact will not
    re-collide with *each other* later.  They still collide with everything
    else, and the alternative -- letting welded neighbours generate contacts
    -- destabilises the solver badly enough to dominate the error.
    """
    n = len(members)
    lows = [
        (
            m.center[0] - 0.5 * m.size[0],
            m.center[1] - 0.5 * m.size[1],
            m.center[2] - 0.5 * m.size[2],
        )
        for m in members
    ]
    highs = [
        (
            m.center[0] + 0.5 * m.size[0],
            m.center[1] + 0.5 * m.size[1],
            m.center[2] + 0.5 * m.size[2],
        )
        for m in members
    ]

    out: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if all(
                min(highs[i][k], highs[j][k]) - max(lows[i][k], lows[j][k]) > tol
                for k in range(3)
            ):
                out.append((members[i].name, members[j].name))
    return tuple(out)


def _slabs_at(x: float, y: float, storey: int, plan: FloorPlan) -> tuple[str, ...]:
    """Names of *every* slab above `storey` that covers (x, y).

    A column standing where four bays meet carries all four slabs, not
    whichever one happens to be enumerated first.  Returning only the first
    match produced slabs held up by a single column, which is structurally
    absurd and made the building impossible to collapse: removing a contiguous
    group of columns left neighbouring slabs still nominally supported.
    """
    return tuple(
        f"slab_s{storey}_b{i}"
        for i, bay in enumerate(plan.bays)
        if bay.contains_point(x, y, tol=_SUPPORT_TOL_M)
    )


def _slab_under(x: float, y: float, storey: int, plan: FloorPlan) -> str | None:
    """The single slab a partition rests on -- walls need only one anchor."""
    slabs = _slabs_at(x, y, storey, plan)
    return slabs[0] if slabs else None
