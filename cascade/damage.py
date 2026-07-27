"""Blast damage: which part of the building the event destroys.

Separated from the simulation itself because it is a *geometric* question with
a deterministic answer, and keeping it pure makes it testable without spinning
up a physics engine.

Why the damage is defined geometrically rather than emerging from a load-driven
cascade -- this was arrived at the expensive way, so it is worth recording:

Every member in the model is rigidly welded to its neighbours, which makes the
intact building effectively a single rigid body.  Remove a column anywhere and
the load redistributes across the entire structure instantly, with no falloff
with distance.  That global coupling makes a *partial* collapse unreachable by
tuning member strength:

* Members weak enough that the damaged bay falls  -> with 13 of 18 ground
  columns left carrying 11 MN, the average survivor sits at 845 kN against a
  772 kN capacity, so every column in the building is over capacity and the
  whole thing comes down.
* Members strong enough that the undamaged frame survives -> the damaged
  section simply hangs from the intact one and nothing falls at all.

Modelling connection failure instead restores locality in principle, but in
practice any disturbance spikes weld forces past their threshold, tearing
connections building-wide and collapsing everything again.

A blast does not politely wait for load to redistribute: it destroys the
columns *and* the connections in its radius, directly.  So that is what this
models.  The collapse is still entirely unscripted -- which members are
damaged is chosen here, but how the freed structure falls, where debris lands,
how slabs pancake and what voids survive underneath is left completely to the
contact solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import CollapseParams
from .rng import stage_rng
from .structure import MemberKind, Structure

#: Member kinds the blast destroys outright rather than merely freeing.
#: Slabs and beams are heavy and survive as falling debris; columns and
#: partitions in the blast radius are gone.
_FRANGIBLE_KINDS = frozenset({MemberKind.COLUMN, MemberKind.WALL})

#: Blast centroids to try before giving up on the size constraints.
_MAX_ATTEMPTS: int = 64

#: Extra reach past a bay's half-diagonal, so the corner columns are safely
#: inside the blast rather than exactly on its boundary, metres.
_CORNER_MARGIN_M: float = 0.6

#: Blast centre jitter about the chosen bay centroid, as a fraction of the
#: nominal blast radius -- enough that seeds differ, small enough that the bay
#: it is aimed at still loses its corners.
_CENTRE_JITTER_FRACTION: float = 0.25


@dataclass(frozen=True, slots=True)
class BlastDamage:
    """What the blast destroyed, as a pure function of geometry and seed."""

    center: tuple[float, float, float]
    radius_m: float
    #: Columns and partitions inside the radius, at every storey: destroyed.
    destroyed_columns: tuple[str, ...]
    #: Members inside the radius that lose their support but survive as debris.
    freed_members: tuple[str, ...]
    #: Connections severed by the blast -- those touching a destroyed column,
    #: and those spanning the boundary between damaged and intact structure.
    severed_welds: tuple[str, ...]

    @property
    def summary(self) -> str:
        return (
            f"blast r={self.radius_m:.1f} m at "
            f"({self.center[0]:.1f}, {self.center[1]:.1f}) -> "
            f"{len(self.destroyed_columns)} columns destroyed, "
            f"{len(self.freed_members)} members freed, "
            f"{len(self.severed_welds)} connections severed"
        )


def _in_radius(
    center: tuple[float, float, float], radius: float, point: tuple[float, float, float]
) -> bool:
    """Membership is horizontal: the blast damages a full-height section."""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return dx * dx + dy * dy <= radius * radius


def compute_damage(
    structure: Structure,
    center: tuple[float, float, float],
    radius_m: float,
) -> BlastDamage:
    """Resolve a blast at `center` of `radius_m` into concrete damage."""
    inside = {
        m.name for m in structure.members if _in_radius(center, radius_m, m.center)
    }
    # A blast destroys what is frangible in its radius -- the columns and the
    # partitions alike -- and merely frees what is not.  Leaving the walls
    # standing was the last thing propping the building up: the deflection
    # head gap is only 25 mm, so a freed slab fell that far, landed squarely
    # on the partitions of the storey below, and stopped.  Slab drop measured
    # 0.11 m against a 3 m storey height.
    destroyed = tuple(
        sorted(
            m.name
            for m in structure.members
            if m.kind in _FRANGIBLE_KINDS and m.name in inside
        )
    )
    destroyed_set = set(destroyed)
    freed = tuple(sorted(inside - destroyed_set))

    severed: list[str] = []
    for weld in structure.welds:
        a_in = weld.body_a in inside
        b_in = weld.body_b in inside if weld.body_b is not None else False
        touches_destroyed = (
            weld.body_a in destroyed_set
            or (weld.body_b is not None and weld.body_b in destroyed_set)
        )
        # A weld to the world counts as "outside", so a freed member's ground
        # fixing is correctly severed.
        crosses_boundary = a_in != b_in
        if touches_destroyed or crosses_boundary:
            severed.append(weld.name)

    return BlastDamage(
        center=center,
        radius_m=radius_m,
        destroyed_columns=destroyed,
        freed_members=freed,
        severed_welds=tuple(sorted(severed)),
    )


def _farthest_corner_distance(
    bay: object, center: tuple[float, float, float]
) -> float:
    """Distance from `center` to the farthest horizontal corner of `bay`."""
    half_x = 0.5 * bay.size[0]
    half_y = 0.5 * bay.size[1]
    return max(
        float(np.hypot(bay.center[0] + sx * half_x - center[0],
                       bay.center[1] + sy * half_y - center[1]))
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
    )


def slab_supports(structure: Structure) -> dict[str, set[str]]:
    """Map each slab to the set of columns bearing directly beneath it."""
    out: dict[str, set[str]] = {}
    for weld in structure.welds:
        if weld.name.startswith("w_head_col") and weld.body_b is not None:
            out.setdefault(weld.body_b, set()).add(weld.body_a)
    return out


def _unsupports_a_slab(
    supports: dict[str, set[str]], destroyed: set[str]
) -> bool:
    """True if some slab has lost every column beneath it."""
    return any(cols and cols <= destroyed for cols in supports.values())


def sample_damage(
    structure: Structure, seed: int, params: CollapseParams
) -> BlastDamage:
    """Sample a blast that damages a worthwhile but survivable fraction.

    Two bounds, both of which matter for the scenario rather than the physics:
    the blast must take out enough of the frame that a real section comes down,
    and it must leave enough of the building standing that there is somewhere
    for robots to search and survivors to be trapped.  A seed that flattens
    everything is not a rescue mission.
    """
    rng = stage_rng(seed, "collapse")
    ground = [
        m
        for m in structure.members
        if m.kind is MemberKind.COLUMN and m.storey == 0
    ]
    bays = [m for m in structure.members if m.kind is MemberKind.SLAB and m.storey == 0]
    if not ground or not bays:
        return compute_damage(structure, (0.0, 0.0, 0.0), 0.0)

    supports = slab_supports(structure)
    total = len(ground)
    lo = max(1, int(round(params.min_destroyed_column_fraction * total)))
    hi = max(lo, int(round(params.max_destroyed_column_fraction * total)))

    best: BlastDamage | None = None
    best_hit = -1
    for _ in range(_MAX_ATTEMPTS):
        # Centre the blast on a bay rather than an arbitrary point.  A slab
        # rests on the four-to-five columns at its corners, so a blast that
        # does not reach all of them leaves the bay standing on the survivors
        # and only strips beams and partitions.  Centred on the bay, a radius
        # of its half-diagonal takes the whole support out and the bay comes
        # down -- which is the difference between a collapse and a scouring.
        bay = bays[int(rng.integers(len(bays)))]
        jitter = params.blast_radius_m * _CENTRE_JITTER_FRACTION
        center = (
            bay.center[0] + float(rng.uniform(-jitter, jitter)),
            bay.center[1] + float(rng.uniform(-jitter, jitter)),
            0.0,
        )
        # Reach to the bay's farthest corner *from the jittered centre*, not
        # to its half-diagonal.  A slab is bay-sized, so it goes on resting by
        # contact on any corner column the blast fails to reach -- welds are
        # irrelevant to that.  Measuring the radius from the nominal centre
        # left the far corner standing whenever the jitter pushed the blast
        # towards the near one, and the floor simply did not fall.
        radius = max(
            params.blast_radius_m,
            _farthest_corner_distance(bay, center) + _CORNER_MARGIN_M,
        )

        damage = compute_damage(structure, center, radius)
        destroyed = set(damage.destroyed_columns)
        hit = sum(1 for c in ground if c.name in destroyed)
        # Both conditions have to hold.  The size bounds keep the scenario
        # usable; the support condition is what makes anything actually fall.
        # A blast can destroy plenty of columns and still leave every slab
        # propped on a survivor, in which case the building is merely scoured
        # of its beams and partitions and no floor comes down at all.
        if lo <= hit <= hi and _unsupports_a_slab(supports, destroyed):
            return damage
        if best is None or abs(hit - lo) < abs(best_hit - lo):
            best, best_hit = damage, hit

    return best if best is not None else compute_damage(structure, (0.0, 0.0, 0.0), 0.0)
