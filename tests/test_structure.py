"""Invariants of the structural graph and the building envelope.

The envelope tests exist because two of the four facades were missing for a
long time and no numeric check noticed: room areas tiled perfectly, member
counts looked sane, and the building was simply open on two sides.  It took
rendering the model to see it.  These assertions are the cheap version of
that observation.
"""

from __future__ import annotations

import pytest

from cascade.floorplan import build_floorplan
from cascade.geometry import EPS_M, Axis
from cascade.params import BuildingParams
from cascade.structure import (
    WALL_HEAD_GAP_M,
    MemberKind,
    build_structure,
    column_nodes,
)

SEEDS = (0, 7, 42, 123, 1000)


@pytest.fixture(scope="module")
def params() -> BuildingParams:
    return BuildingParams()


def _built(seed: int, params: BuildingParams):
    plan = build_floorplan(seed, params)
    return plan, build_structure(plan)


# --- envelope ----------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_every_facade_has_walls(seed: int, params: BuildingParams) -> None:
    """All four faces must be enclosed, on every storey.

    An open-sided building vents: the stack effect that makes the stairwell a
    chimney needs an enclosure, and the RSSI model's wall raycasts need
    something to cross.
    """
    plan, structure = _built(seed, params)
    footprint = plan.footprint
    walls = structure.of_kind(MemberKind.WALL)

    facades = {
        "y0": (Axis.Y, footprint.y0),
        "y1": (Axis.Y, footprint.y1),
        "x0": (Axis.X, footprint.x0),
        "x1": (Axis.X, footprint.x1),
    }
    for storey in range(params.n_storeys):
        on_storey = [w for w in walls if w.storey == storey]
        for label, (axis, coordinate) in facades.items():
            if axis is Axis.Y:
                found = [
                    w
                    for w in on_storey
                    if abs(w.center[1] - coordinate) <= 0.5 * params.wall_thickness_m
                    + EPS_M
                ]
            else:
                found = [
                    w
                    for w in on_storey
                    if abs(w.center[0] - coordinate) <= 0.5 * params.wall_thickness_m
                    + EPS_M
                ]
            assert found, f"seed {seed} storey {storey}: facade {label} has no walls"


@pytest.mark.parametrize("seed", SEEDS)
def test_facades_are_substantially_closed(seed: int, params: BuildingParams) -> None:
    """Each facade must be mostly wall, not a token segment.

    Guards the corridor-cutting bug specifically: subtracting the wrong axis's
    interval punched holes at meaningless positions and left facades in
    tatters while still passing a "some wall exists" check.
    """
    plan, structure = _built(seed, params)
    footprint = plan.footprint
    walls = [w for w in structure.of_kind(MemberKind.WALL) if w.storey == 1]
    half_t = 0.5 * params.wall_thickness_m + EPS_M

    for axis, coordinate, span in (
        (Axis.Y, footprint.y0, footprint.width),
        (Axis.Y, footprint.y1, footprint.width),
        (Axis.X, footprint.x0, footprint.height),
        (Axis.X, footprint.x1, footprint.height),
    ):
        if axis is Axis.Y:
            on_face = [w for w in walls if abs(w.center[1] - coordinate) <= half_t]
            covered = sum(w.size[0] for w in on_face)
        else:
            on_face = [w for w in walls if abs(w.center[0] - coordinate) <= half_t]
            covered = sum(w.size[1] for w in on_face)
        assert covered >= 0.5 * span, (
            f"seed {seed}: facade at {axis.value}={coordinate} is only "
            f"{covered:.1f} m of {span:.1f} m closed"
        )


# --- exits -------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_exits_exist_and_sit_on_the_envelope(seed: int, params: BuildingParams) -> None:
    plan, _ = _built(seed, params)
    footprint = plan.footprint
    assert plan.exits, f"seed {seed} has no exits; nowhere to lead a survivor"
    for door in plan.exits:
        if door.axis is Axis.X:
            assert abs(door.at - footprint.x0) <= EPS_M or (
                abs(door.at - footprint.x1) <= EPS_M
            )
        else:
            assert abs(door.at - footprint.y0) <= EPS_M or (
                abs(door.at - footprint.y1) <= EPS_M
            )


@pytest.mark.parametrize("seed", SEEDS)
def test_exits_are_not_all_on_one_facade(seed: int, params: BuildingParams) -> None:
    """Two doors side by side give the swarm a single egress direction."""
    plan, _ = _built(seed, params)
    if len(plan.exits) < 2:
        pytest.skip("only one exit placed for this seed")
    facades = {(d.axis.value, round(d.at, 6)) for d in plan.exits}
    assert len(facades) > 1, f"seed {seed}: every exit is on the same facade"


# --- members and welds -------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_partitions_clear_the_slab_soffit(seed: int, params: BuildingParams) -> None:
    """The deflection head gap must actually be there.

    Without it partitions become load-bearing columns and hold the building up
    through a collapse.
    """
    _, structure = _built(seed, params)
    expected = params.storey_height_m - params.slab_thickness_m - WALL_HEAD_GAP_M
    for wall in structure.of_kind(MemberKind.WALL):
        assert wall.size[2] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_columns_stop_below_the_slab(seed: int, params: BuildingParams) -> None:
    """Columns must not interpenetrate the slab they carry."""
    _, structure = _built(seed, params)
    expected = params.storey_height_m - params.slab_thickness_m
    for column in structure.of_kind(MemberKind.COLUMN):
        assert column.size[2] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_every_weld_references_a_real_member(seed: int, params: BuildingParams) -> None:
    _, structure = _built(seed, params)
    names = {m.name for m in structure.members}
    for weld in structure.welds:
        assert weld.body_a in names
        assert weld.body_b is None or weld.body_b in names


@pytest.mark.parametrize("seed", SEEDS)
def test_ground_columns_are_anchored(seed: int, params: BuildingParams) -> None:
    _, structure = _built(seed, params)
    anchored = {
        w.body_a for w in structure.welds if w.body_b is None and "col_s0" in w.body_a
    }
    ground = {
        m.name
        for m in structure.members
        if m.kind is MemberKind.COLUMN and m.storey == 0
    }
    assert ground <= anchored, "some ground column is not fixed to the world"


@pytest.mark.parametrize("seed", SEEDS)
def test_column_nodes_are_deduplicated(seed: int, params: BuildingParams) -> None:
    plan, _ = _built(seed, params)
    nodes = column_nodes(plan.bays)
    assert len(nodes) == len({(round(x, 6), round(y, 6)) for x, y in nodes})


def test_member_mass_is_positive(params: BuildingParams) -> None:
    _, structure = _built(7, params)
    for member in structure.members:
        assert member.mass_kg > 0.0
        assert member.thermal_mass_j_per_k > 0.0
        assert member.yield_threshold_n > 0.0


# --- stairs ------------------------------------------------------------------


def _stair_extent(member) -> tuple[float, float]:
    """World z of a member's lowest and highest corners."""
    import numpy as np

    half = np.array(member.half_extents, dtype=np.float64)
    w, x, y, z = member.quat
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    reach = float(np.abs(rot[2, :]) @ half)
    return member.center[2] - reach, member.center[2] + reach


@pytest.mark.parametrize("seed", SEEDS)
def test_ground_flights_do_not_dip_below_grade(
    seed: int, params: BuildingParams
) -> None:
    """The one thing that made every staircase in the building unusable.

    A flight authored with its top face meeting the floor puts its *underside*
    a thickness below that floor.  Over a punched-out slab that is fine; on
    solid ground the collapse resolves the penetration by ejecting the whole
    flight upward, and it settles with its head a full thickness proud of the
    landing it is supposed to meet -- an unclimbable ledge at the top of the
    stairs, which is exactly where a robot is pitched up on its rear wheels.
    """
    _, structure = _built(seed, params)
    for member in structure.members:
        if member.kind is not MemberKind.STAIR or "_s0_" not in member.name:
            continue
        low, _ = _stair_extent(member)
        assert low >= -EPS_M, f"{member.name} starts {-low:.3f} m below grade"


@pytest.mark.parametrize("seed", SEEDS)
def test_each_flight_meets_the_landing_it_serves(
    seed: int, params: BuildingParams
) -> None:
    """A flight's top face and its landing's top face are the same height."""
    _, structure = _built(seed, params)
    by_name = {m.name: m for m in structure.members}
    for name, member in by_name.items():
        if "_f0" not in name or name.endswith("_threshold"):
            continue
        landing = by_name.get(name.rsplit("_f0", 1)[0] + "_landing")
        if landing is None:
            continue
        _, flight_top = _stair_extent(member)
        _, landing_top = _stair_extent(landing)
        assert flight_top == pytest.approx(landing_top, abs=1e-6)
