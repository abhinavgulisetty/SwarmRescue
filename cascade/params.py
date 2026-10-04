"""Generation parameters for the Cascade world generator.

Every tunable lives here as a named constant or a frozen dataclass field.  No
magic numbers are permitted downstream: if a number has meaning, it is named
here and carries a unit in its comment.

All dataclasses are frozen.  Variants are produced with `dataclasses.replace`,
never by mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

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


class RoomKind(StrEnum):
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
    #: Ground-floor exits through the envelope.  These are the egress terminal
    #: states of the routing problem, so a building without them is not a
    #: scenario -- there is nowhere to lead a survivor to.
    n_exits: int = 2
    exit_width_m: float = 1.8
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
        _require(
            self.n_storeys >= 2,
            "n_storeys must be >= 2 for a stairwell to matter",
        )
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
        _require(self.n_exits >= 1, "a building needs at least one exit")
        _require(self.exit_width_m > 0.0, "exit_width_m must be positive")

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
            0.0
            < self.min_destroyed_column_fraction
            <= self.max_destroyed_column_fraction
            < 1.0,
            "destroyed-column fractions must satisfy 0 < min <= max < 1",
        )
        _require(self.timestep_s > 0.0, "timestep_s must be positive")

    @property
    def n_settle_steps(self) -> int:
        return round(self.settle_seconds / self.timestep_s)


@dataclass(frozen=True, slots=True)
class TraversabilityParams:
    """Stage 4 -- what the robot can actually drive over.

    The slope limit describes a **tracked** micro-robot, not a wheeled one,
    and that is a deliberate spec decision rather than a tuned constant.  The
    building's stairs are pitched at 34 degrees; a platform limited to ~22
    degrees cannot use them, so the traversability graph would come out
    disconnected between storeys and the mission would be impossible by
    construction.  Tracked rescue platforms in this size class routinely climb
    35 degrees, so that is what the swarm is.

    The step limit is the opposite kind of number: it is measured off the
    Phase 0 chassis, which is a four-wheel skid-steer and not a tracked one.
    Where the two disagree the graph takes the pessimistic value, because a
    route the demo robot cannot drive is worse than no route.
    """

    #: Horizontal raster resolution.  Finer than the 0.4 m the proposal quotes
    #: for the robots' own belief octree, because this grid is *ground truth*
    #: and has to resolve a doorway.  Conservative rasterization inflates every
    #: obstacle by half a cell, so at 0.4 m a 0.9 m door shrinks to a single
    #: free voxel and the planner cannot tell a doorway from a wall.
    voxel_size_m: float = 0.25
    #: Vertical raster resolution.  Much finer than the horizontal cell,
    #: because a 0.4 m voxel cannot resolve whether a 0.18 m step is climbable.
    voxel_height_m: float = 0.10
    #: Maximum vertical *discontinuity* the platform can mount, metres.
    #:
    #: A step, not a grade: the two are measured separately (see
    #: `extract._step_between`) because the platform's limits on them are not
    #: the same number and conflating them is what made the graph unusable.
    #: A rigid four-wheeler mounts roughly its wheel radius driven square at a
    #: ledge and rather less at an angle, so this is set just above the wheel
    #: radius to cover the residual lip at the foot of a stair flight and the
    #: few centimetres a landing settles by, and no more.
    max_step_height_m: float = 0.13
    #: Maximum traversable slope.
    max_slope_deg: float = 35.0
    #: A gap narrower than this cannot be squeezed through.
    min_squeeze_width_m: float = 0.30
    #: Vertical clearance the robot needs to occupy a cell (its own height).
    min_clearance_m: float = 0.28
    #: Radius of the disc the robot sweeps when it turns on the spot, metres.
    #:
    #: The graph is planned in *configuration space*: obstacles are grown by
    #: this before standable cells are extracted, so a node means "the robot's
    #: centre can be here", not "this point is not inside a wall".  Without it
    #: the planner is solving for a point robot -- it routed a 0.34 x 0.24 m
    #: chassis through 0.25 m gaps, and the robot wedged itself on geometry the
    #: planner had certified as open.  Half the chassis diagonal is the right
    #: value because the platform is skid-steered and yaws while driving, so no
    #: smaller disc is guaranteed clear.
    robot_radius_m: float = 0.21

    def __post_init__(self) -> None:
        _require(self.voxel_size_m > 0.0, "voxel_size_m must be positive")
        _require(self.voxel_height_m > 0.0, "voxel_height_m must be positive")
        _require(self.max_step_height_m > 0.0, "max_step_height_m must be positive")
        _require(0.0 < self.max_slope_deg < 90.0, "max_slope_deg must be in (0, 90)")
        _require(self.min_clearance_m > 0.0, "min_clearance_m must be positive")
        _require(self.robot_radius_m > 0.0, "robot_radius_m must be positive")

    @property
    def clearance_levels(self) -> int:
        """Free vertical cells needed above a standable surface."""
        import math

        return max(1, math.ceil(self.min_clearance_m / self.voxel_height_m))


@dataclass(frozen=True, slots=True)
class CascadeParams:
    """Top-level bundle handed to `cascade.generate`."""

    building: BuildingParams = field(default_factory=BuildingParams)
    collapse: CollapseParams = field(default_factory=CollapseParams)
    traversability: TraversabilityParams = field(default_factory=TraversabilityParams)


def compact_demo_params() -> CascadeParams:
    """A legible, fast disaster preset for the end-to-end showpiece.

    The research-scale world remains the default.  This variant uses fewer,
    larger bays so losing one supported plate reads as a section collapse on
    screen instead of as a couple of tiny fragments in a wide building shot.
    """

    return CascadeParams(
        building=BuildingParams(
            n_storeys=3,
            footprint_m=(14.0, 10.0),
            bay_depth=3,
            room_depth=1,
            min_bay_area_m2=16.0,
            min_room_area_m2=7.0,
            min_room_side_m=2.0,
            max_room_aspect=3.2,
            n_exits=2,
            exit_width_m=1.6,
            n_stairwells=1,
            n_shafts=1,
            n_ducts=0,
            stairwell_size_m=(4.0, 2.8),
            shaft_size_m=(1.8, 1.8),
            duct_size_m=(1.2, 1.2),
        ),
        collapse=CollapseParams(
            blast_radius_m=2.2,
            settle_seconds=6.0,
            min_destroyed_column_fraction=0.15,
            max_destroyed_column_fraction=0.40,
        ),
    )


def _require(condition: bool, message: str) -> None:
    """Fail fast at the construction boundary with a readable message."""
    if not condition:
        raise ValueError(message)
