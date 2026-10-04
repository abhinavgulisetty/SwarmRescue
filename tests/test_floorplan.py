"""Invariants the floor-plan grammar must hold for every seed.

These are the properties later stages silently depend on: rooms must tile
their bay exactly (the voxel raster assumes it), connectors must align
vertically (the stack effect assumes it), and every room must be reachable
(the mission assumes it).
"""

from __future__ import annotations

import pytest

from cascade.floorplan import (
    _carve_connector,
    _connector_size,
    _mst_edges,
    build_floorplan,
    subdivide_bay,
)
from cascade.geometry import Rect, shared_wall
from cascade.params import BuildingParams, RoomKind
from cascade.rng import stage_rng

#: Seeds spanning the development range; kept small so the suite stays fast.
SEEDS = (0, 7, 42, 123, 199)


@pytest.fixture(scope="module")
def params() -> BuildingParams:
    return BuildingParams()


def _plans(seeds=SEEDS, params: BuildingParams | None = None):
    p = params or BuildingParams()
    return [(s, build_floorplan(s, p)) for s in seeds]


def _find_component(parent: list[int], node: int) -> int:
    """Find a component root while compressing the union-find path."""
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _union_components(parent: list[int], left: int, right: int) -> None:
    """Join two components in the test's small connectivity oracle."""
    parent[_find_component(parent, right)] = _find_component(parent, left)


# --- determinism -------------------------------------------------------------


def test_generation_is_deterministic(params: BuildingParams) -> None:
    """The same seed must reproduce the plan exactly -- frozen dataclasses
    compare structurally, so this is a true bit-level equality check."""
    assert build_floorplan(7, params) == build_floorplan(7, params)


def test_distinct_seeds_give_distinct_plans(params: BuildingParams) -> None:
    assert build_floorplan(7, params) != build_floorplan(8, params)


# --- tiling ------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_bays_tile_the_footprint_exactly(seed: int, params: BuildingParams) -> None:
    fp = build_floorplan(seed, params)
    total = sum(b.area for b in fp.bays)
    assert total == pytest.approx(fp.footprint.area, rel=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_bays_do_not_overlap(seed: int, params: BuildingParams) -> None:
    fp = build_floorplan(seed, params)
    for i, a in enumerate(fp.bays):
        for b in fp.bays[i + 1 :]:
            assert not a.overlaps(b, tol=1e-6)


@pytest.mark.parametrize("seed", SEEDS)
def test_rooms_tile_every_storey_exactly(seed: int, params: BuildingParams) -> None:
    """Room area must sum to the footprint on each storey: the connector
    guillotine is the easiest thing here to get subtly wrong."""
    fp = build_floorplan(seed, params)
    for storey in fp.storeys:
        total = sum(r.rect.area for r in storey.rooms)
        assert total == pytest.approx(fp.footprint.area, rel=1e-9), (
            f"storey {storey.index} does not tile the footprint"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_rooms_do_not_overlap(seed: int, params: BuildingParams) -> None:
    fp = build_floorplan(seed, params)
    for storey in fp.storeys:
        rects = [r.rect for r in storey.rooms]
        for i, a in enumerate(rects):
            for b in rects[i + 1 :]:
                assert not a.overlaps(b, tol=1e-6)


def test_connector_guillotine_tiles_its_host_cell(params: BuildingParams) -> None:
    rng = stage_rng(3, "floorplan")
    cell = Rect(0.0, 0.0, 6.0, 5.0)
    for kind in (RoomKind.STAIRWELL, RoomKind.SHAFT, RoomKind.DUCT):
        width, depth = _connector_size(kind, params)
        shaft, remainder = _carve_connector(cell, width, depth, rng)
        pieces = (shaft, *remainder)
        assert sum(p.area for p in pieces) == pytest.approx(cell.area, rel=1e-9)
        for i, a in enumerate(pieces):
            for b in pieces[i + 1 :]:
                assert not a.overlaps(b, tol=1e-6)


# --- connectors --------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_connectors_are_vertically_aligned(seed: int, params: BuildingParams) -> None:
    """A connector must occupy an identical footprint on every storey, or it
    is a stack of holes rather than a chimney."""
    fp = build_floorplan(seed, params)
    for plan in fp.connectors:
        for storey in fp.storeys:
            matching = [
                r for r in storey.rooms if r.kind is plan.kind and r.bay == plan.bay
            ]
            assert len(matching) == 1, (
                f"{plan.kind} missing or duplicated on storey {storey.index}"
            )
            assert matching[0].rect == plan.rect


@pytest.mark.parametrize("seed", SEEDS)
def test_every_building_has_a_stairwell(seed: int, params: BuildingParams) -> None:
    fp = build_floorplan(seed, params)
    assert len(fp.stairwells()) >= 1


@pytest.mark.parametrize("seed", SEEDS)
def test_connectors_are_sized_to_function(seed: int, params: BuildingParams) -> None:
    """A duct must not silently inherit a whole room's footprint.

    The bound is deliberately asymmetric.  Oversizing is the bug this guards
    (a connector swallowing its host cell); undersizing is a legitimate
    outcome when no cell in the chosen bay can fit the full footprint, so it
    only has to stay above a usable floor.
    """
    fp = build_floorplan(seed, params)
    for plan in fp.connectors:
        target = params.connector_target_area_m2(plan.kind)
        area = plan.rect.area
        assert area <= 1.05 * target, (
            f"{plan.kind.value} oversized: {area:.1f} m2 vs target {target:.1f}"
        )
        assert area >= 0.5 * target, (
            f"{plan.kind.value} unusably small: {area:.1f} m2 vs target {target:.1f}"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_connectors_occupy_distinct_bays(seed: int, params: BuildingParams) -> None:
    fp = build_floorplan(seed, params)
    bays = [c.bay for c in fp.connectors]
    assert len(bays) == len(set(bays))


# --- connectivity ------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_every_room_is_reachable(seed: int, params: BuildingParams) -> None:
    """Doors plus corridors must leave one connected component per storey."""
    fp = build_floorplan(seed, params)
    for storey in fp.storeys:
        n = len(storey.rooms)
        parent = list(range(n))

        for door in storey.doors:
            _union_components(parent, door.room_a, door.room_b)
        for leg in storey.corridors:
            touched = [i for i, r in enumerate(storey.rooms) if r.rect.overlaps(leg)]
            for other in touched[1:]:
                _union_components(parent, touched[0], other)

        components = {_find_component(parent, i) for i in range(n)}
        assert len(components) == 1, (
            f"storey {storey.index} has {len(components)} disconnected components"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_doors_lie_on_a_real_shared_wall(seed: int, params: BuildingParams) -> None:
    fp = build_floorplan(seed, params)
    for storey in fp.storeys:
        for door in storey.doors:
            a = storey.rooms[door.room_a].rect
            b = storey.rooms[door.room_b].rect
            assert shared_wall(a, b, params.door_width_m) is not None
            # The opening must sit inside the wall segment, not past its end.
            lo, hi = door.wall.lo, door.wall.hi
            along = door.center[1] if door.wall.axis.value == "x" else door.center[0]
            assert lo - 1e-9 <= along <= hi + 1e-9


# --- the corridor spine ------------------------------------------------------


def test_mst_spans_all_terminals() -> None:
    points = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0), (2.5, 2.5)]
    edges = _mst_edges(points)
    assert len(edges) == len(points) - 1
    reached = {0}
    # A spanning tree must connect every terminal to the root.
    for _ in range(len(points)):
        for a, b in edges:
            if a in reached or b in reached:
                reached |= {a, b}
    assert reached == set(range(len(points)))


def test_mst_of_a_single_point_is_empty() -> None:
    assert _mst_edges([(0.0, 0.0)]) == ()


@pytest.mark.parametrize("seed", SEEDS)
def test_storeys_differ_but_share_the_structural_grid(
    seed: int, params: BuildingParams
) -> None:
    """Partitions vary per storey; the bay grid does not."""
    fp = build_floorplan(seed, params)
    layouts = {tuple(sorted((r.rect for r in s.rooms), key=repr)) for s in fp.storeys}
    assert len(layouts) > 1, "every storey has an identical layout"


# --- parameter validation ----------------------------------------------------


def test_invalid_params_fail_fast() -> None:
    with pytest.raises(ValueError):
        BuildingParams(n_storeys=1)
    with pytest.raises(ValueError):
        BuildingParams(split_jitter=0.7)
    with pytest.raises(ValueError):
        BuildingParams(door_width_m=99.0)


def test_subdivide_bay_respects_minimum_area(params: BuildingParams) -> None:
    rng = stage_rng(5, "floorplan")
    bay = Rect(0.0, 0.0, 10.0, 8.0)
    for cell in subdivide_bay(bay, params, rng):
        assert cell.area >= params.min_room_area_m2 - 1e-9
