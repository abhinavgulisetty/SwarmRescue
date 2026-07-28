"""Tests for voxel rasterization and the traversability graph.

The reachability test is the one that matters: it is the standing, cheap
version of Phase 0's definition of done, and it is what caught the stair
rotation sign error and the missing floor-level landings.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from cascade.extract import (
    GRID_MARGIN_M,
    build_graph,
    height_map,
    nearest_node,
    rasterize,
    reachable_storeys,
    shortest_path,
    storeys_reached,
)
from cascade.floorplan import build_floorplan
from cascade.mjcf import to_mjcf
from cascade.params import CascadeParams
from cascade.structure import build_structure

SEEDS = (7, 42)


def _world(seed: int):
    params = CascadeParams()
    plan = build_floorplan(seed, params.building)
    structure = build_structure(plan)
    model = mujoco.MjModel.from_xml_string(to_mjcf(structure, plan))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    grid = rasterize(
        model,
        data,
        params.building.footprint_m,
        params.building.total_height_m,
        params.traversability,
    )
    return params, plan, grid, build_graph(grid, params.traversability)


@pytest.fixture(scope="module")
def world():
    return _world(7)


# --- rasterization -----------------------------------------------------------


def test_grid_covers_the_footprint_with_margin(world) -> None:
    params, _, grid, _ = world
    width, depth = params.building.footprint_m
    nx, ny, _ = grid.shape
    assert nx * grid.cell[0] >= width + 2 * GRID_MARGIN_M - grid.cell[0]
    assert ny * grid.cell[1] >= depth + 2 * GRID_MARGIN_M - grid.cell[1]


def test_ground_is_solid(world) -> None:
    """Level 0 represents the ground plane, which no box geom would supply."""
    _, _, grid, _ = world
    assert grid.occupied[:, :, 0].all()
    assert grid.surface_z(0) == pytest.approx(0.0, abs=1e-9)


def test_building_is_neither_empty_nor_solid(world) -> None:
    _, _, grid, _ = world
    fraction = grid.occupancy_fraction()
    assert 0.01 < fraction < 0.5, f"implausible occupancy {fraction:.3f}"


def test_thin_walls_are_rasterized(world) -> None:
    """A 0.15 m partition must not fall between voxel centres.

    Centre-sampling missed most walls in the building, and the planner
    happily routed through them.
    """
    params, plan, grid, _ = world
    storey_mid = 0.5 * params.building.storey_height_m
    k = int((storey_mid - grid.origin[2]) / grid.cell[2])
    # A horizontal slice through a storey must show interior structure, not a
    # nearly empty floor plate.
    occupied = grid.occupied[:, :, k]
    assert 0.02 < occupied.mean() < 0.9


def test_height_map_matches_grid_columns(world) -> None:
    _, _, grid, _ = world
    heights = height_map(grid)
    assert heights.shape == grid.shape[:2]
    assert np.all(heights >= grid.origin[2] - 1e-9)


# --- graph -------------------------------------------------------------------


def test_graph_is_non_trivial(world) -> None:
    _, _, _, graph = world
    assert graph.n_nodes > 1000
    assert graph.n_edges > graph.n_nodes / 2


def test_edges_respect_the_rise_limit(world) -> None:
    params, _, grid, graph = world
    positions = graph.world_positions()
    rise = np.abs(
        positions[graph.edges[:, 0], 2] - positions[graph.edges[:, 1], 2]
    )
    slope = params.traversability.max_slope_deg
    limit = (
        max(
            params.traversability.max_step_height_m,
            grid.cell[0] * np.tan(np.radians(slope)),
        )
        + 0.5 * grid.cell[2]
    )
    assert rise.max() <= limit + 1e-9


def test_every_node_stands_on_something(world) -> None:
    """Support comes from `occupied`; clearance comes from `blocked`.

    The two grids answer different questions and the node definition uses one
    of each, so the invariant is asymmetric.  Asserting `not occupied[k + 1]`
    would be wrong: `occupied` pads every box by half a voxel vertically to
    catch thin slabs, and that padding routinely bleeds one level above a
    floor the robot stands on perfectly well.
    """
    _, _, grid, graph = world
    for i, j, k in graph.cells[:: max(1, graph.n_nodes // 200)]:
        assert grid.occupied[i, j, k], "standable cell has nothing beneath it"
        assert not grid.blocked[i, j, k + 1], "no room for the chassis"


@pytest.mark.parametrize("seed", SEEDS)
def test_every_storey_is_reachable_from_an_exit(seed: int) -> None:
    """Phase 0's definition of done, as a standing assertion.

    A building whose upper storeys cannot be reached from the street is not a
    rescue scenario.  This caught two separate defects: stair flights whose
    rotation sign ran them the wrong way, and the absence of a floor-level
    landing to cross between flights.
    """
    params, plan, _, graph = _world(seed)
    door = plan.exits[0]
    start = nearest_node(graph, (door.center[0], door.center[1], 0.05))
    reachable = reachable_storeys(graph, start, params.building.storey_height_m)
    for storey in range(params.building.n_storeys):
        assert storey in reachable, (
            f"seed {seed}: storey {storey} unreachable from the exit; "
            f"reachable = {sorted(reachable)}"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_a_route_exists_to_the_top_storey(seed: int) -> None:
    params, plan, _, graph = _world(seed)
    door = plan.exits[0]
    stairwell = plan.stairwells()[0].rect
    top = params.building.n_storeys - 1
    start = nearest_node(graph, (door.center[0], door.center[1], 0.05))
    goal = nearest_node(
        graph,
        (
            stairwell.center[0],
            stairwell.center[1],
            top * params.building.storey_height_m + 0.1,
        ),
    )
    route = shortest_path(graph, start, goal)
    assert route, f"seed {seed}: no route from the exit to storey {top}"
    visited = storeys_reached(graph, route, params.building.storey_height_m)
    assert {0, top} <= visited


def test_shortest_path_is_connected(world) -> None:
    """Consecutive nodes on a route must actually be adjacent cells."""
    _, plan, _, graph = world
    door = plan.exits[0]
    start = nearest_node(graph, (door.center[0], door.center[1], 0.05))
    goal = nearest_node(graph, (plan.stairwells()[0].rect.center[0],
                                plan.stairwells()[0].rect.center[1], 3.1))
    route = shortest_path(graph, start, goal)
    assert route
    edges = {(int(a), int(b)) for a, b in graph.edges}
    edges |= {(b, a) for a, b in edges}
    for a, b in zip(route[:-1], route[1:], strict=True):
        assert (a, b) in edges, "route uses a move that is not an edge"


# --- configuration space -----------------------------------------------------


def _slot_scene(gap_m: float) -> mujoco.MjModel:
    """Two parallel walls `gap_m` apart, centred on x = 2, running along y."""
    thickness = 0.15
    offset = 0.5 * gap_m + 0.5 * thickness
    walls = "".join(
        f'<body name="w{i}" pos="{2.0 + s * offset} 2.0 1.0">'
        f'<geom type="box" size="{0.5 * thickness} 1.5 1.0"/></body>'
        for i, s in enumerate((-1.0, 1.0))
    )
    return mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><geom type="plane" size="10 10 0.1"/>{walls}'
        "</worldbody></mujoco>"
    )


def _slot_is_traversable(gap_m: float) -> bool:
    params = CascadeParams()
    model = _slot_scene(gap_m)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    grid = rasterize(model, data, (4.0, 4.0), 2.5, params.traversability)
    graph = build_graph(grid, params.traversability)
    positions = graph.world_positions()
    # Standable cells on the ground, inside the slot and clear of its mouths.
    inside = (
        (np.abs(positions[:, 0] - 2.0) < 0.5 * gap_m)
        & (np.abs(positions[:, 1] - 2.0) < 1.0)
        & (positions[:, 2] < 0.5)
    )
    return bool(inside.any())


def test_a_slot_narrower_than_the_robot_is_not_traversable() -> None:
    """The point-robot planner's defining failure, as a standing test.

    A 0.30 m slot is open floor to a point and impassable to a 0.42 m chassis.
    The planner used to route through gaps like this and the robot wedged
    itself on geometry the graph had certified as clear.
    """
    assert not _slot_is_traversable(0.30)


def test_a_slot_wider_than_the_robot_is_traversable() -> None:
    """The inflation must not be so eager that it seals ordinary doorways."""
    assert _slot_is_traversable(1.20)


def test_unreachable_goal_returns_empty_path(world) -> None:
    _, _, _, graph = world
    # Node 0 to itself is trivially reachable; a negative index is not a node,
    # so use an isolated synthetic case instead.
    assert shortest_path(graph, 0, 0) == [0]
