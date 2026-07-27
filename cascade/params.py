"""Generation parameters for the Cascade world generator.

Every tunable lives here as a named constant or a frozen dataclass field.  No
magic numbers are permitted downstream: if a number has meaning, it is named
here and carries a unit in its comment.

All dataclasses are frozen.  Variants are produced with `dataclasses.replace`,
never by mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# --- Physical constants -------------------------------------------------------

GRAVITY_M_S2: float = 9.81
CONCRETE_DENSITY_KG_M3: float = 2400.0
#: Partition walls are lightweight board, not structural concrete.
PARTITION_DENSITY_KG_M3: float = 700.0

# --- Discretization -----------------------------------------------------------

#: Edge length of a traversability voxel, in metres (proposal SS4.2 stage 4).
VOXEL_SIZE_M: float = 0.4
#: Heightfield raster resolution, in metres per cell.
HFIELD_CELL_M: float = 0.15


class RoomKind(str, Enum):
    """Semantic room class.

    The kind drives fuel load (hazard, phase 1) and traversal semantics, so it
    is part of the floor plan rather than a later annotation.
    """

    OFFICE = "office"
    LOBBY = "lobby"
    CORRIDOR = "corridor"
    STAIRWELL = "stairwell"
    SHAFT = "shaft"
    DUCT = "duct"

    @property
    def is_vertical_connector(self) -> bool:
        """True for kinds that connect storeys -- and act as chimneys."""
        return self in _VERTICAL_CONNECTOR_KINDS


_VERTICAL_CONNECTOR_KINDS = frozenset(
    {RoomKind.STAIRWELL, RoomKind.SHAFT, RoomKind.DUCT}
)

#: Fuel load per room kind, kg of equivalent combustible per square metre.
#: An office is not a stairwell -- this is what makes the fire front
#: anisotropic and its shape informative about the building (proposal SS4.3a).
FUEL_LOAD_KG_M2: dict[RoomKind, float] = {
    RoomKind.OFFICE: 35.0,
    RoomKind.LOBBY: 12.0,
    RoomKind.CORRIDOR: 8.0,
    RoomKind.STAIRWELL: 2.0,
    RoomKind.SHAFT: 1.0,
    RoomKind.DUCT: 1.0,
}


@dataclass(frozen=True, slots=True)
class BuildingParams:
    """Floor-plan grammar and structural sizing."""

    n_storeys: int = 4
    storey_height_m: float = 3.0
    footprint_m: tuple[float, float] = (24.0, 18.0)

    # -- BSP partition (stage 1) --
    #: Recursion depth for the shared structural bay grid (<= 2**depth bays).
    bay_depth: int = 3
    #: Further per-storey recursion inside each bay (<= 2**depth rooms/bay).
    room_depth: int = 2
    min_bay_area_m2: float = 26.0
    min_room_area_m2: float = 11.0
    min_room_side_m: float = 2.4
    #: Reject a split whose child would exceed this width:height ratio.
    max_room_aspect: float = 2.8
    #: Splits land in [0.5 - j, 0.5 + j] of the parent extent.
    split_jitter: float = 0.16

    # -- Openings --
    door_width_m: float = 0.9
    #: P(door) on a shared wall decays as exp(-len / this), so long shared
    #: walls between big rooms are not riddled with doorways.
    door_prob_length_scale_m: float = 6.0
    min_doors_per_room: int = 1

    # -- Member sizing --
    wall_thickness_m: float = 0.15
    slab_thickness_m: float = 0.20
    column_side_m: float = 0.35
    beam_width_m: float = 0.25
    beam_depth_m: float = 0.35

    # -- Vertical connectors --
    n_stairwells: int = 1
    n_shafts: int = 1
    n_ducts: int = 1
    #: Target footprint per connector kind, metres.  A connector claims the
    #: subdivided cell whose area is closest to this, so a stairwell is a
    #: stairwell rather than a whole structural bay.
    stairwell_size_m: tuple[float, float] = (4.5, 3.0)
    shaft_size_m: tuple[float, float] = (2.2, 2.2)
    duct_size_m: tuple[float, float] = (1.5, 1.5)

    def connector_target_area_m2(self, kind: RoomKind) -> float:
        """Target plan area for a vertical connector of `kind`."""
        sizes = {
            RoomKind.STAIRWELL: self.stairwell_size_m,
            RoomKind.SHAFT: self.shaft_size_m,
            RoomKind.DUCT: self.duct_size_m,
        }
        if kind not in sizes:
            raise ValueError(f"{kind} is not a vertical connector kind")
        w, d = sizes[kind]
        return w * d

    def __post_init__(self) -> None:
        _require(self.n_storeys >= 2, "n_storeys must be >= 2 for a stairwell to matter")
        _require(self.storey_height_m > 0.0, "storey_height_m must be positive")
        _require(all(d > 0.0 for d in self.footprint_m), "footprint must be positive")
        _require(0.0 <= self.split_jitter < 0.5, "split_jitter must be in [0, 0.5)")
        _require(self.max_room_aspect >= 1.0, "max_room_aspect must be >= 1")
        _require(
            self.min_room_side_m**2 <= self.min_room_area_m2,
            "min_room_area_m2 must admit a square room of min_room_side_m",
        )
        _require(
            self.door_width_m < self.min_room_side_m,
            "door_width_m must fit inside the shortest admissible wall",
        )
        _require(
            self.min_bay_area_m2 >= 2.0 * self.min_room_area_m2,
            "a bay must be splittable into at least two rooms",
        )
        _require(self.bay_depth >= 1 and self.room_depth >= 1, "depths must be >= 1")

    @property
    def total_height_m(self) -> float:
        return self.n_storeys * self.storey_height_m


@dataclass(frozen=True, slots=True)
class CollapseParams:
    """Stage 3 -- collapse by simulation, not by hand."""

    #: Radius of the blast, metres.  Everything within this horizontal
    #: distance of the centroid is damaged, at every storey, so the damage is
    #: a contiguous full-height section rather than scattered points.
    blast_radius_m: float = 6.5
    #: Wall-clock physics simulated after release, seconds (proposal: 5-10 s).
    settle_seconds: float = 8.0
    #: Integrator step, seconds.  Small enough for stiff slab-on-slab contact.
    timestep_s: float = 0.002
    #: Settling is declared early once total kinetic energy stays under this
    #: threshold (joules) for `quiescent_ticks`.
    quiescent_kinetic_energy_j: float = 5.0
    quiescent_ticks: int = 250

    # -- blast extent --
    #: Bounds on how much of the ground frame the blast may take out, as a
    #: fraction of ground columns.  Too little and no section comes down; too
    #: much and there is no building left to search, which is not a rescue
    #: scenario.
    min_destroyed_column_fraction: float = 0.15
    max_destroyed_column_fraction: float = 0.40

    def __post_init__(self) -> None:
        _require(self.settle_seconds > 0.0, "settle_seconds must be positive")
        _require(self.blast_radius_m > 0.0, "blast_radius_m must be positive")
        _require(
            0.0 < self.min_destroyed_column_fraction
            <= self.max_destroyed_column_fraction < 1.0,
            "destroyed-column fractions must satisfy 0 < min <= max < 1",
        )
        _require(self.timestep_s > 0.0, "timestep_s must be positive")

    @property
    def n_settle_steps(self) -> int:
        return int(round(self.settle_seconds / self.timestep_s))


@dataclass(frozen=True, slots=True)
class TraversabilityParams:
    """Stage 4 -- what a small ground robot can actually drive over."""

    voxel_size_m: float = VOXEL_SIZE_M
    #: Maximum climbable vertical discontinuity between adjacent voxels.
    max_step_height_m: float = 0.12
    #: Maximum traversable slope.
    max_slope_deg: float = 22.0
    #: A gap narrower than this cannot be squeezed through.
    min_squeeze_width_m: float = 0.30
    #: Vertical clearance a robot needs to occupy a voxel.
    min_clearance_m: float = 0.25

    def __post_init__(self) -> None:
        _require(self.voxel_size_m > 0.0, "voxel_size_m must be positive")
        _require(0.0 < self.max_slope_deg < 90.0, "max_slope_deg must be in (0, 90)")


@dataclass(frozen=True, slots=True)
class CascadeParams:
    """Top-level bundle handed to `cascade.generate`."""

    building: BuildingParams = field(default_factory=BuildingParams)
    collapse: CollapseParams = field(default_factory=CollapseParams)
    traversability: TraversabilityParams = field(default_factory=TraversabilityParams)


def _require(condition: bool, message: str) -> None:
    """Fail fast at the construction boundary with a readable message."""
    if not condition:
        raise ValueError(message)
