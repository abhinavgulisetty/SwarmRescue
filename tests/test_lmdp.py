"""Contracts for the Phase 1 linearly-solvable egress router."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from cascade.extract import TraversabilityGraph, VoxelGrid, shortest_path
from ember.routing import (
    LMDPConfig,
    OracleCostConfig,
    build_lmdp,
    greedy_route,
    oracle_state_cost,
    solve_jacobi,
    trajectory_optimism,
)
from hazard import HazardDomain, HazardState


def _graph(
    n_nodes: int,
    edges: list[tuple[int, int]],
    edge_cost: list[float] | None = None,
) -> TraversabilityGraph:
    """Make a tiny graph whose nodes lie at consecutive hazard-cell centres."""

    cells = np.column_stack(
        (
            np.arange(n_nodes, dtype=np.int32),
            np.zeros(n_nodes, dtype=np.int32),
            np.zeros(n_nodes, dtype=np.int32),
        )
    )
    occupied = np.ones((n_nodes, 1, 1), dtype=bool)
    grid = VoxelGrid(
        origin=(0.0, 0.0, 0.0),
        cell=(1.0, 1.0, 1.0),
        occupied=occupied,
        blocked=np.zeros_like(occupied),
    )
    return TraversabilityGraph(
        grid=grid,
        cells=cells,
        surface_z=np.full(n_nodes, 0.5),
        edges=np.asarray(edges, dtype=np.int32).reshape(-1, 2),
        edge_cost=np.asarray(
            edge_cost if edge_cost is not None else [1.0] * len(edges),
            dtype=np.float64,
        ),
    )


def _row(values: np.ndarray, row_ptr: np.ndarray, node: int) -> np.ndarray:
    return values[int(row_ptr[node]) : int(row_ptr[node + 1])]


@pytest.mark.parametrize(
    "overrides",
    [
        {"control_temperature": 0.0},
        {"passive_cost_scale": -1.0},
        {"self_loop_weight": -1.0},
        {"minimum_state_cost": 0.0},
        {"tolerance": 0.0},
        {"max_sweeps": 0},
        {"max_sweeps": 1.5},
        {"control_temperature": 1.0e308},
    ],
    ids=[
        "zero-control-temperature",
        "negative-passive-scale",
        "negative-self-loop",
        "zero-minimum-cost",
        "zero-tolerance",
        "zero-sweeps",
        "fractional-sweeps",
        "unrepresentable-contraction",
    ],
)
def test_config_rejects_invalid_solver_parameters(
    overrides: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError):
        replace(LMDPConfig(), **overrides)


def test_build_enforces_strictly_positive_nonterminal_costs() -> None:
    graph = _graph(3, [(0, 1), (1, 2)])
    config = LMDPConfig(minimum_state_cost=0.125)

    model = build_lmdp(
        graph,
        state_cost=np.array([0.0, 0.0, 99.0]),
        exit_nodes=(2,),
        config=config,
    )

    assert np.all(model.state_cost[~model.exit_mask & ~model.death_mask] >= 0.125)
    assert model.state_cost[2] == 0.0


def test_passive_rows_are_normalized_and_prefer_easier_edges() -> None:
    graph = _graph(3, [(0, 1), (0, 2)], edge_cost=[1.0, 4.0])
    model = build_lmdp(
        graph,
        state_cost=np.full(3, 0.1),
        exit_nodes=(2,),
        config=LMDPConfig(self_loop_weight=0.0),
    )

    for node in range(graph.n_nodes):
        assert _row(model.passive_prob, model.row_ptr, node).sum() == pytest.approx(
            1.0
        )

    columns = _row(model.col_idx, model.row_ptr, 0)
    probabilities = _row(model.passive_prob, model.row_ptr, 0)
    probability_by_node = dict(
        zip(columns.tolist(), probabilities.tolist(), strict=True)
    )
    assert probability_by_node[1] > probability_by_node[2]


def test_jacobi_matches_the_analytic_three_node_chain() -> None:
    graph = _graph(3, [(0, 1), (1, 2)])
    cost = 0.4
    temperature = 0.8
    config = LMDPConfig(
        control_temperature=temperature,
        self_loop_weight=0.0,
        tolerance=1.0e-12,
        max_sweeps=2_000,
    )
    model = build_lmdp(
        graph,
        state_cost=np.array([cost, cost, 0.0]),
        exit_nodes=(2,),
        config=config,
    )

    solution = solve_jacobi(model)

    attenuation = np.exp(-cost / temperature)
    expected_z1 = attenuation / (2.0 - attenuation**2)
    expected = np.array([attenuation * expected_z1, expected_z1, 1.0])
    np.testing.assert_allclose(solution.desirability, expected, rtol=0.0, atol=2e-11)
    np.testing.assert_allclose(
        solution.value[:2],
        -temperature * np.log(expected[:2]),
        rtol=0.0,
        atol=2e-11,
    )
    assert solution.converged
    assert 0 < solution.sweeps <= config.max_sweeps
    assert solution.residual <= config.tolerance
    assert solution.relative_residual <= config.tolerance


def test_jacobi_reports_when_the_sweep_budget_is_exhausted() -> None:
    graph = _graph(3, [(0, 1), (1, 2)])
    config = LMDPConfig(
        self_loop_weight=0.0,
        tolerance=1.0e-15,
        max_sweeps=1,
    )
    model = build_lmdp(
        graph,
        state_cost=np.array([0.1, 0.1, 0.0]),
        exit_nodes=(2,),
        config=config,
    )

    solution = solve_jacobi(model)

    assert not solution.converged
    assert solution.sweeps == config.max_sweeps
    assert solution.residual > config.tolerance


def test_jacobi_marks_an_exact_final_budget_sweep_converged() -> None:
    graph = _graph(2, [(0, 1)])
    config = LMDPConfig(
        self_loop_weight=0.0,
        tolerance=1.0e-15,
        max_sweeps=1,
    )
    model = build_lmdp(
        graph,
        state_cost=np.array([0.1, 0.0]),
        exit_nodes=(1,),
        config=config,
    )

    solution = solve_jacobi(model)

    assert solution.converged
    assert solution.sweeps == 1
    assert solution.residual == 0.0


def test_jacobi_waits_for_a_tiny_exit_signal_to_cross_the_graph() -> None:
    graph = _graph(
        5,
        [(0, 1), (1, 2), (2, 3), (2, 4)],
        edge_cost=[1.0, 1.0, 30.0, 0.0],
    )
    model = build_lmdp(
        graph,
        state_cost=np.array([0.1, 0.1, 0.1, 0.0, 0.1]),
        exit_nodes=(3,),
        config=LMDPConfig(self_loop_weight=0.0),
    )

    solution = solve_jacobi(model)

    assert solution.converged
    assert np.all(solution.desirability[:3] > 0.0)
    assert greedy_route(model, solution, 0)[-1] == 3


def test_exit_and_death_are_exact_absorbing_boundaries() -> None:
    graph = _graph(3, [(0, 1), (0, 2)])
    model = build_lmdp(
        graph,
        state_cost=np.full(3, 0.2),
        exit_nodes=(1,),
        death_nodes=(2,),
    )

    solution = solve_jacobi(model)

    assert solution.desirability[1] == 1.0
    assert solution.value[1] == 0.0
    assert solution.desirability[2] == 0.0
    assert np.isposinf(solution.value[2])


def test_optimal_policy_rows_are_probability_distributions() -> None:
    graph = _graph(4, [(0, 1), (0, 2), (1, 3), (2, 3)])
    model = build_lmdp(
        graph,
        state_cost=np.array([0.1, 0.2, 0.5, 0.0]),
        exit_nodes=(3,),
    )

    solution = solve_jacobi(model)

    assert np.all(np.isfinite(solution.policy_prob))
    assert np.all(solution.policy_prob >= 0.0)
    for node in (0, 1, 2):
        assert _row(
            solution.policy_prob, model.row_ptr, node
        ).sum() == pytest.approx(1.0)


def test_greedy_route_is_deterministic_and_reaches_the_exit() -> None:
    graph = _graph(3, [(0, 1), (1, 2)])
    model = build_lmdp(
        graph,
        state_cost=np.array([0.1, 0.1, 0.0]),
        exit_nodes=(2,),
    )
    solution = solve_jacobi(model)

    first = greedy_route(model, solution, start=0)
    second = greedy_route(model, solution, start=0)

    assert first == [0, 1, 2]
    assert second == first


def test_greedy_route_rejects_a_high_probability_dead_end() -> None:
    # Passive dynamics strongly prefer cheap edge 0-2, so a raw policy argmax
    # bounces 0<->2.  Positive state cost guarantees that the route to egress
    # has a strictly increasing desirability field even when its passive edge
    # is difficult.
    graph = _graph(
        4,
        [(0, 1), (1, 3), (0, 2)],
        edge_cost=[5.0, 1.0, 0.0],
    )
    model = build_lmdp(
        graph,
        state_cost=np.array([0.1, 0.1, 0.1, 0.0]),
        exit_nodes=(3,),
        config=LMDPConfig(self_loop_weight=0.0),
    )
    solution = solve_jacobi(model)
    row = model.row(0)
    policy = dict(
        zip(
            model.col_idx[row].tolist(),
            solution.policy_prob[row].tolist(),
            strict=True,
        )
    )

    assert policy[2] > policy[1]
    assert solution.desirability[2] < solution.desirability[0]
    assert greedy_route(model, solution, start=0) == [0, 1, 3]


def test_high_hazard_selects_long_safe_route_over_shortest_path() -> None:
    # 0-1-4 is the two-edge chimney; 0-2-3-4 is the safe detour.
    graph = _graph(5, [(0, 1), (1, 4), (0, 2), (2, 3), (3, 4)])
    assert shortest_path(graph, 0, 4) == [0, 1, 4]
    model = build_lmdp(
        graph,
        state_cost=np.array([0.05, 20.0, 0.05, 0.05, 0.0]),

        exit_nodes=(4,),
        config=LMDPConfig(self_loop_weight=0.0),
    )

    route = greedy_route(model, solve_jacobi(model), start=0)

    assert route == [0, 2, 3, 4]


def test_oracle_cost_samples_hazard_state_at_graph_nodes() -> None:
    graph = _graph(3, [(0, 1), (1, 2)])
    active = np.ones((1, 1, 3), dtype=bool)
    domain = HazardDomain(
        origin_m=(0.0, 0.0, 0.0),
        cell_size_m=1.0,
        active=active,
        connector=np.zeros_like(active),
        vertical_open=np.zeros_like(active),
        initial_fuel_kg_m3=np.zeros(active.shape),
    )
    state = HazardState(
        temperature_k=np.array([[[300.0, 375.0, 500.0]]]),
        fuel_kg_m3=np.zeros(active.shape),
        smoke_kg_m3=np.array([[[0.0, 0.1, 0.4]]]),
    )

    first = oracle_state_cost(graph, domain, state, OracleCostConfig())
    second = oracle_state_cost(graph, domain, state, OracleCostConfig())

    assert first.shape == (3,)
    assert np.all(np.isfinite(first))
    assert np.all(first > 0.0)
    assert first[0] < first[1] < first[2]
    np.testing.assert_array_equal(first, second)


def test_oracle_cost_treats_graph_margin_as_clear_ambient() -> None:
    graph = _graph(4, [(0, 1), (1, 2), (2, 3)])
    active = np.ones((1, 1, 3), dtype=bool)
    domain = HazardDomain(
        origin_m=(0.0, 0.0, 0.0),
        cell_size_m=1.0,
        active=active,
        connector=np.zeros_like(active),
        vertical_open=np.zeros_like(active),
        initial_fuel_kg_m3=np.zeros(active.shape),
    )
    state = HazardState(
        temperature_k=np.array([[[300.0, 300.0, 800.0]]]),
        fuel_kg_m3=np.zeros(active.shape),
        smoke_kg_m3=np.array([[[0.0, 0.0, 1.0]]]),
    )
    config = OracleCostConfig()

    cost = oracle_state_cost(graph, domain, state, config)

    assert cost[2] > config.base_step_cost
    assert cost[3] == pytest.approx(config.base_step_cost)


def test_disconnected_component_has_zero_desirability_and_no_route() -> None:
    graph = _graph(4, [(0, 1), (2, 3)])
    model = build_lmdp(
        graph,
        state_cost=np.array([0.2, 0.2, 0.2, 0.0]),
        exit_nodes=(3,),
        config=LMDPConfig(self_loop_weight=0.0),
    )

    solution = solve_jacobi(model)

    np.testing.assert_array_equal(solution.desirability[:2], np.zeros(2))
    assert np.all(np.isposinf(solution.value[:2]))
    assert greedy_route(model, solution, start=0) == []


def test_build_and_solve_do_not_mutate_inputs_and_are_reproducible() -> None:
    graph = _graph(4, [(0, 1), (1, 2), (2, 3)])
    state_cost = np.array([0.3, 0.2, 0.1, 0.0])
    original_cost = state_cost.copy()
    original_edges = graph.edges.copy()
    original_edge_cost = graph.edge_cost.copy()
    config = LMDPConfig(tolerance=1.0e-11)

    first_model = build_lmdp(graph, state_cost, exit_nodes=(3,), config=config)
    first = solve_jacobi(first_model)
    second_model = build_lmdp(graph, state_cost, exit_nodes=(3,), config=config)
    second = solve_jacobi(second_model)

    np.testing.assert_array_equal(state_cost, original_cost)
    np.testing.assert_array_equal(graph.edges, original_edges)
    np.testing.assert_array_equal(graph.edge_cost, original_edge_cost)
    np.testing.assert_array_equal(first_model.row_ptr, second_model.row_ptr)
    np.testing.assert_array_equal(first_model.col_idx, second_model.col_idx)
    np.testing.assert_array_equal(first_model.passive_prob, second_model.passive_prob)
    np.testing.assert_array_equal(first.desirability, second.desirability)
    np.testing.assert_array_equal(first.policy_prob, second.policy_prob)
    assert first.sweeps == second.sweeps
    assert first.residual == second.residual

def test_trajectory_optimism_is_zero_for_deterministic_chain() -> None:
    graph = _graph(2, [(0, 1)])
    model = build_lmdp(
        graph,
        np.array((0.2, 0.0)),
        exit_nodes=(1,),
        config=LMDPConfig(self_loop_weight=0.0),
    )
    solution = solve_jacobi(model)

    diagnostic = trajectory_optimism(model, solution)

    assert diagnostic.gap[0] == pytest.approx(0.0, abs=1.0e-9)


def test_trajectory_optimism_is_positive_for_stochastic_cost_paths() -> None:
    graph = _graph(
        4,
        [(0, 1), (0, 2), (1, 3), (2, 3)],
    )
    model = build_lmdp(
        graph,
        np.array((0.1, 0.1, 1.5, 0.0)),
        exit_nodes=(3,),
        config=LMDPConfig(self_loop_weight=0.0),
    )
    solution = solve_jacobi(model)

    diagnostic = trajectory_optimism(model, solution)

    assert diagnostic.gap[0] > 0.0
    assert diagnostic.mean_gap > 0.0
