"""Acceptance and terminal-output behaviour of the Phase 0 demo."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cascade import demo
from cascade.cinema import SwarmCamera, Trails
from cascade.damage import sample_damage
from cascade.demo import (
    DemoResult,
    Stage,
    _append_hold,
    _disaster_mission_status,
    _fan_out_disaster_assignments,
    _interior_depths,
    _local_disaster_assignment,
    _mission_status,
    _routing_hazard_center,
    _thermal_damage_active,
    _thermal_damage_radius,
    _thermal_failure_zone,
    _trail_surface_positions,
)
from cascade.extract import TraversabilityGraph, VoxelGrid
from cascade.floorplan import build_floorplan
from cascade.params import BuildingParams, CascadeParams, compact_demo_params
from cascade.robot import RobotSpec, RobotState
from cascade.structure import build_structure
from cascade.swarm import Assignment


def _result(*statuses: str) -> DemoResult:
    per_robot = tuple(
        (f"robot{i}", i + 1, i + 1, status) for i, status in enumerate(statuses)
    )
    return DemoResult(
        seed=7,
        n_robots=len(statuses),
        storeys_reached={0, 1, 2, 3, 4},
        planner_reachable={1, 2, 3, 4},
        per_robot=per_robot,
        steps=100,
        frames=10,
        video=Path("demo.mp4"),
    )


def test_acceptance_requires_every_robot_to_arrive() -> None:
    assert _result("arrived", "arrived", "arrived", "arrived").succeeded
    assert not _result("arrived", "fallen", "arrived", "arrived").succeeded
    assert not _result("arrived", "driving", "arrived", "arrived").succeeded


def test_cli_returns_failure_for_a_bad_run(monkeypatch) -> None:
    monkeypatch.setattr(demo, "run_demo", lambda **_: _result("arrived", "fallen"))
    assert demo.main(["--no-render"]) == 1


def test_cli_returns_success_for_an_accepted_run(monkeypatch) -> None:
    monkeypatch.setattr(demo, "run_demo", lambda **_: _result("arrived", "arrived"))
    assert demo.main(["--no-render"]) == 0


def test_rescue_requires_every_physical_survivor_to_be_evacuated():
    result = replace(
        _result("arrived", "arrived", "arrived"),
        evacuation_enabled=True,
        live_disaster=True,
        distributed_physical=True,
        failed_members=("beam",),
        survivors_hidden=3,
        survivors_localized=3,
        survivors_contacted=3,
        survivors_evacuated=3,
        routing_converged=True,
        geometry_revisions=1,
        peer_factors=1,
        robot_interior_depth_m=(-1.0, -1.0, -1.0),
    )
    assert result.accepted
    assert not replace(result, survivors_evacuated=2).accepted
    assert not replace(result, survivors_contacted=2).accepted
    assert not replace(result, survivors_localized=2).accepted
    assert not replace(result, geometry_revisions=0).accepted
    assert not replace(result, robot_interior_depth_m=(-1.0, 0.0, -1.0)).accepted


@pytest.mark.parametrize(
    "args",
    [["--evacuate"], ["--mission-horizon-s", "nan"], ["--mission-horizon-s", "601"]],
)
def test_bad_rescue_configuration_rejected_before_scene_construction(args):
    with pytest.raises(SystemExit) as raised:
        demo.main(args)
    assert raised.value.code == 2


def test_compact_cli_requires_the_live_disaster_mode() -> None:
    with pytest.raises(SystemExit) as raised:
        demo.main(["--compact", "--no-render"])
    assert raised.value.code == 2


def test_disaster_acceptance_proves_multiple_robots_moved() -> None:
    result = replace(
        _result("arrived", "arrived", "arrived", "arrived"),
        live_disaster=True,
        failed_members=("column_1",),
        robot_progress=(0.2, 0.1, 0.0, 0.0),
        robot_displacement_m=(0.6, 0.5, 0.7, 0.8),
        robot_interior_depth_m=(2.0, 1.8, 1.9, 2.1),
        belief_reroutes=4,
        collapse_replans=4,
        routing_converged=True,
        routing_cost_before=2.0,
        routing_cost_after=1.0,
        survivors_hidden=4,
        survivors_localized=4,
        max_localization_error_m=0.5,
        autonomous_search=True,
    )

    assert result.accepted
    assert not replace(result, robot_displacement_m=(0.6, 0.1, 0.1, 0.0)).accepted
    assert not replace(result, robot_interior_depth_m=(2.0, 0.1, -0.1, -0.2)).accepted
    assert not replace(
        result,
        per_robot=(*result.per_robot[:3], (*result.per_robot[3][:3], "driving")),
    ).accepted
    assert not replace(result, failed_members=()).accepted
    assert not replace(result, belief_reroutes=0).accepted
    assert not replace(result, belief_reroutes=3).accepted
    assert not replace(result, collapse_replans=3).accepted
    assert replace(result, collapse_replans=3, collapse_retained=1).accepted
    assert not replace(result, collapse_replans=1, collapse_retained=3).accepted
    assert not replace(result, routing_converged=False).accepted
    assert not replace(result, routing_cost_after=2.0).accepted
    assert not replace(result, autonomous_search=False).accepted
    assert not replace(result, survivors_localized=3).accepted
    assert not replace(result, max_localization_error_m=2.1).accepted


def test_local_disaster_route_runs_from_spawn_to_building_entry() -> None:
    spec = RobotSpec()
    assignment = Assignment(
        spec=spec,
        spawn=(3.0, 4.0, spec.ride_height_m),
        yaw_rad=0.0,
        route=np.array([[1.0, 4.0, 0.0], [0.5, 4.0, 0.0]]),
        goal_storey=1,
    )

    local = _local_disaster_assignment(assignment)

    assert np.allclose(
        local.route,
        ((3.0, 4.0, 0.0), (1.0, 4.0, 0.0), (0.5, 4.0, 0.0)),
    )
    assert local.goal_storey == 0


def test_routing_hazard_sits_on_the_shared_entry_route() -> None:
    spec = RobotSpec()
    route = np.column_stack(
        (np.arange(12, dtype=np.float64), np.ones(12), np.zeros(12))
    )
    assignment = Assignment(spec, (0.0, 1.0, 0.2), 0.0, route, 0)

    centre = _routing_hazard_center([assignment])

    assert centre[:2] == tuple(route[10, :2])


def test_interior_depth_is_positive_only_beyond_the_facade() -> None:
    footprint = build_floorplan(7, BuildingParams()).footprint
    positions = np.array(((2.0, 2.0, 0.2), (24.5, 2.0, 0.2)))

    assert _interior_depths(positions, footprint) == (2.0, -0.5)


def test_local_disaster_goals_fan_out_to_distinct_certified_cells() -> None:
    cells = np.array(((8, 8, 0), (8, 9, 0), (8, 7, 0), (7, 8, 0), (9, 8, 0), (7, 9, 0)))
    grid = VoxelGrid(
        origin=(0.0, 0.0, -0.1),
        cell=(0.25, 0.25, 0.1),
        occupied=np.zeros((10, 10, 1), dtype=bool),
        blocked=np.zeros((10, 10, 1), dtype=bool),
    )
    graph = TraversabilityGraph(
        grid=grid,
        cells=cells,
        surface_z=np.zeros(6),
        edges=np.array(((0, 1), (0, 2), (0, 3), (0, 4), (0, 5))),
        edge_cost=np.full(5, 0.75),
    )
    anchor = graph.world_positions()[0]
    assignments = [
        Assignment(
            spec=replace(RobotSpec(), name=f"robot{index}"),
            spawn=(3.0, 2.0, RobotSpec().ride_height_m),
            yaw_rad=0.0,
            route=np.array((anchor,)),
            goal_storey=0,
        )
        for index in range(4)
    ]

    fanned = _fan_out_disaster_assignments(
        assignments, graph, build_floorplan(7, BuildingParams()).footprint
    )

    assert len({tuple(assignment.route[-1]) for assignment in fanned}) == 4


def test_terminal_frame_is_held_for_the_requested_duration() -> None:
    class Writer:
        def __init__(self) -> None:
            self.frames = []

        def append(self, frame) -> None:
            self.frames.append(frame)

    writer = Writer()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    _append_hold(writer, frame, fps=12, seconds=2.0)
    assert len(writer.frames) == 24
    assert all(item is frame for item in writer.frames)


def test_trails_use_surface_height_like_planned_routes() -> None:
    spec = RobotSpec()
    swarm = SimpleNamespace(
        members=[SimpleNamespace(assignment=SimpleNamespace(spec=spec))]
    )
    chassis = np.array([[1.0, 2.0, 3.0]])
    surface = _trail_surface_positions(chassis, swarm)
    assert surface[0, 2] == 3.0 - spec.ride_height_m
    assert chassis[0, 2] == 3.0, "the caller's positions were mutated"


def _stage() -> Stage:
    building = BuildingParams()
    return Stage(
        seed=7,
        building=building,
        groups=SimpleNamespace(),
        camera=SwarmCamera(footprint_m=building.footprint_m),
        trails=Trails(n_robots=2),
        aspect=1.6,
    )


def test_rescue_terminal_card_counts_survivors_instead_of_robot_queue_arrivals(
    monkeypatch,
):
    stage = _stage()
    stage.evacuation = SimpleNamespace(
        contacted=8,
        rescued=8,
        evacuation_time_s=np.arange(8, dtype=float) + 50,
    )
    stage.firmware_robots = 2
    captured = {}

    def card(frame, **values):
        captured.update(values)
        return frame

    monkeypatch.setattr(demo, "research_card", card)
    stage._results_card(
        np.zeros((20, 20, 3), np.uint8),
        [SimpleNamespace(status="driving") for _ in range(8)],
    )
    assert captured["title"] == "8/8 SURVIVORS EVACUATED"
    assert "centralized BLE" not in captured["footer"]
    assert captured["metrics"][0].value == "8/8"


def test_storey_slice_switches_only_when_it_can_show_the_subject() -> None:
    stage = _stage()

    below = stage._slice(2.94)
    above = stage._slice(2.96)
    assert below[0] == 0
    assert above[0] == 1
    for height, (_, _, floor, ceiling) in ((2.94, below), (2.96, above)):
        assert floor <= height < ceiling


def _member(name: str, *, state: RobotState = RobotState.DRIVING):
    return SimpleNamespace(
        name=name,
        state=state,
        status=state.value,
        driver=SimpleNamespace(finished=False, stalled=False),
    )


def test_camera_holds_a_shot_but_recovery_preempts_it() -> None:
    stage = _stage()
    first = _member("robot0")
    second = _member("robot1")

    class FakeSwarm:
        def __init__(self) -> None:
            self.members = [first, second]
            self.preferred = first

        def focus(self):
            return self.preferred

    swarm = FakeSwarm()
    assert stage._choose_subject(swarm, 0.0) is first
    swarm.preferred = second
    assert stage._choose_subject(swarm, 1.0) is first

    second.state = RobotState.RECOVERING
    second.status = second.state.value
    stage.camera._lookat = np.ones(3)
    assert stage._choose_subject(swarm, 1.1) is second
    assert stage.camera._lookat is None, "a camera cut retained the old aim point"


def test_disaster_camera_looks_low_at_the_failure_bay() -> None:
    stage = _stage()
    stage.disaster_focus = (24.0, 0.0, 1.4)
    stage.disaster_at_s = 1.5

    view = stage._disaster_view(np.array((4.0, 4.0, 0.2)))

    assert view.lookat[2] < stage.building.storey_height_m
    assert view.distance <= 1.2 * max(stage.building.footprint_m)
    assert stage._disaster_subtitle(0.8).startswith("STEP 1 OF 4")


def test_live_collapse_waits_for_the_swarm_to_clear_the_entrance() -> None:
    assert not _thermal_damage_active(5999, 0.002)
    assert _thermal_damage_active(6000, 0.002)


def test_compact_disaster_heats_a_larger_fraction_of_its_bay() -> None:
    default = CascadeParams()
    compact = compact_demo_params()

    assert _thermal_damage_radius(default) == pytest.approx(
        0.7 * default.collapse.blast_radius_m
    )
    assert _thermal_damage_radius(compact) == pytest.approx(
        1.2 * compact.collapse.blast_radius_m
    )


def test_disaster_subtitle_explains_the_response_sequence() -> None:
    stage = _stage()
    stage.disaster_focus = (24.0, 0.0, 1.4)
    stage.disaster_at_s = 1.5
    stage.search_status = "4/4 LOCALIZED"
    assert stage._disaster_subtitle(1.6).startswith("STEP 2 OF 4")
    stage.collapse_event_s = 2.0
    stage.collapse_replans = 4
    assert "DAMAGE INTEGRAL" in stage._disaster_subtitle(2.5)
    assert "4 NEW ROUTES" in stage._disaster_subtitle(6.0)


def test_disaster_action_camera_contains_the_local_swarm() -> None:
    positions = np.array([[0.0, 0.0, 0.3], [4.0, 0.0, 0.3]])

    view = Stage._swarm_view(positions, 0.0)

    assert view.lookat[:2] == (2.0, 0.0)
    assert view.distance >= 11.0


def test_terminal_mission_status_is_explicit() -> None:
    complete = SimpleNamespace(members=[_member("robot0", state=RobotState.ARRIVED)])
    failed = SimpleNamespace(members=[_member("robot0", state=RobotState.FALLEN)])

    assert _mission_status(complete, timed_out=False).startswith("MISSION COMPLETE")
    assert "ROBOT0 FALLEN" in _mission_status(failed, timed_out=False)
    assert "TIME LIMIT" in _mission_status(failed, timed_out=True)
    swarm = SimpleNamespace(
        members=[
            _member("robot0", state=RobotState.ARRIVED),
            _member("robot1", state=RobotState.ARRIVED),
        ]
    )
    assert _disaster_mission_status(swarm, 23, 2, 2, 2, 0).startswith(
        "ENTRY MISSION COMPLETE · 2/2 INSIDE"
    )


def test_thermal_failure_targets_members_that_survived_the_first() -> None:
    params = CascadeParams()
    structure = build_structure(build_floorplan(7, params.building))
    first = sample_damage(structure, 7, params.collapse)
    collapse = SimpleNamespace(damage=first)

    second = _thermal_failure_zone(
        structure,
        collapse,
        0.7 * params.collapse.blast_radius_m,
    )

    assert set(second.destroyed_columns) - set(first.destroyed_columns)
