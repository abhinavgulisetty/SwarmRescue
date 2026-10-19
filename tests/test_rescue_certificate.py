"""Exhaustive schedule enumeration independent of the sliding-max recurrence."""

import itertools

import numpy as np
import pytest

from ember.rescue_certificate import certify_integer_rescue, certify_rescue


def _simulate(hazard, approach, approach_ticks, escort, escort_ticks, delay, dt):
    robot = [approach[0]] * delay
    for node, duration in zip(approach[:-1], approach_ticks, strict=True):
        robot.extend([node] * duration)
    survivor = [approach[-1]] * len(robot)
    for node, duration in zip(escort[:-1], escort_ticks, strict=True):
        robot.extend([node] * duration)
        survivor.extend([node] * duration)
    return tuple(
        sum(float(hazard[tick, node]) * dt for tick, node in enumerate(path))
        for path in (robot, survivor)
    )


def _exhaustive(scenarios, approach, durations, escort, lower, upper, delay, dt):
    best = [0.0, 0.0]
    for scenario, timings in itertools.product(
        scenarios,
        itertools.product(
            *(range(lo, hi + 1) for lo, hi in zip(lower, upper, strict=True))
        ),
    ):
        doses = _simulate(scenario, approach, durations, escort, timings, delay, dt)
        best = [max(current, dose) for current, dose in zip(best, doses, strict=True)]
    return best


def _check_witnesses(result, scenarios, approach, durations, escort, delay, dt):
    for agent, witness in enumerate((result.robot_witness, result.survivor_witness)):
        doses = _simulate(
            scenarios[witness.scenario_index],
            approach,
            durations,
            escort,
            witness.escort_ticks,
            delay,
            dt,
        )
        assert doses[agent] == pytest.approx(witness.exposure, rel=1e-12, abs=1e-12)
        assert witness.completion_tick == delay + sum(durations) + sum(
            witness.escort_ticks
        )


def test_complete_prefix_includes_both_waits_and_all_approach_segments():
    hazards = np.tile(np.array([[[2.0, 3.0, 5.0, 1000.0]]]), (1, 12, 1))
    result = certify_rescue(
        hazards,
        (0, 1, 2),
        (2, 3),
        approach_ticks=(2, 1),
        escort_min_ticks=3,
        departure_delay=2,
        dt=0.5,
        robot_budget=13,
        survivor_budget=20,
    )
    assert result.pickup_tick == 5
    assert result.robot_bound == 13
    assert result.survivor_bound == 20
    assert result.feasible
    assert result.earliest_completion_tick == result.latest_completion_tick == 8
    assert result.schedule_count == 1
    # Node 3 is the safe destination: its later field does not enter exposure.
    _check_witnesses(result, hazards, (0, 1, 2), (2, 1), (2, 3), 2, 0.5)


def test_scenario_is_fixed_across_whole_rescue_and_actor_witnesses_may_differ():
    hazards = np.zeros((2, 4, 3))
    hazards[0, 0, 0], hazards[0, 1, 1] = 5, 1
    hazards[1, 0, 1], hazards[1, 1, 1] = 7, 2
    result = certify_rescue(hazards, (0, 1), (1, 2), robot_budget=6, survivor_budget=8)
    assert result.robot_bound == 6
    assert result.survivor_bound == 9
    assert not result.feasible
    assert result.robot_witness.scenario_index == 0
    assert result.survivor_witness.scenario_index == 1
    # Resetting the source between approach and escort would incorrectly give 7.
    assert result.robot_bound < 5 + 2


def test_maximum_lag_is_not_always_the_worst_schedule():
    hazards = np.zeros((1, 6, 4))
    hazards[0, 2, 2] = 10
    result = certify_rescue(
        hazards, (0, 1), (1, 2, 3), escort_min_ticks=(1, 1), escort_max_ticks=(3, 2)
    )
    assert result.survivor_bound == 10
    assert result.survivor_witness.escort_ticks[0] == 1
    assert _simulate(hazards[0], (0, 1), (1,), (1, 2, 3), (3, 2), 0, 1)[1] == 0


def test_deque_dp_matches_exhaustive_all_small_integer_hazards():
    # Includes a moving single-tick peak and ties, across every binary field.
    for flat in itertools.product((0, 1), repeat=8):
        scenarios = np.array(flat, dtype=float).reshape(1, 4, 2)
        result = certify_rescue(
            scenarios, (0,), (0, 1, 0), escort_min_ticks=(1, 1), escort_max_ticks=(2, 2)
        )
        expected = _exhaustive(scenarios, (0,), (), (0, 1, 0), (1, 1), (2, 2), 0, 1)
        assert [result.robot_bound, result.survivor_bound] == expected
        assert result.schedule_count == 4
        assert result.dp_cells == 6  # widths 1, 2, 3
        _check_witnesses(result, scenarios, (0,), (), (0, 1, 0), 0, 1)


def test_random_repeated_nodes_varying_durations_match_direct_enumeration():
    rng = np.random.default_rng(501)
    for _ in range(50):
        approach, escort = (0, 2, 1), (1, 2, 1, 3)
        durations = (int(rng.integers(1, 3)), int(rng.integers(1, 3)))
        lower = tuple(map(int, rng.integers(1, 3, 3)))
        upper = tuple(lo + int(rng.integers(0, 3)) for lo in lower)
        delay, dt = int(rng.integers(0, 3)), 0.25
        horizon = delay + sum(durations) + sum(upper)
        scenarios = rng.uniform(0, 3, (3, horizon, 4))
        result = certify_rescue(
            scenarios,
            approach,
            escort,
            approach_ticks=durations,
            escort_min_ticks=lower,
            escort_max_ticks=upper,
            departure_delay=delay,
            dt=dt,
        )
        expected = _exhaustive(
            scenarios, approach, durations, escort, lower, upper, delay, dt
        )
        np.testing.assert_allclose(
            [result.robot_bound, result.survivor_bound],
            expected,
            atol=1e-12,
            rtol=1e-12,
        )
        _check_witnesses(result, scenarios, approach, durations, escort, delay, dt)
        assert result.numeric_scratch_bytes > sum(upper) * 8


def test_singleton_zero_horizon_and_nominal_dwell_modes():
    empty_time = np.zeros((2, 0, 1))
    zero = certify_rescue(empty_time, (0,), (0,), robot_budget=0, survivor_budget=0)
    assert zero.feasible and zero.robot_bound == zero.survivor_bound == 0
    assert zero.robot_witness.escort_ticks == ()
    assert zero.schedule_count == 1
    hazards = np.ones((1, 5, 3))
    implicit = certify_rescue(hazards, (0, 1), (1, 2), escort_min_ticks=2)
    explicit = certify_rescue(
        hazards, (0, 1), (1, 2), escort_min_ticks=2, escort_max_ticks=2
    )
    assert implicit == explicit


@pytest.mark.parametrize(
    "updates",
    [
        {"escort_max_ticks": 5},  # latest completion would exceed scenario horizon
        {"escort_min_ticks": 2, "escort_max_ticks": 1},
        {"approach_ticks": ()},
        {"escort_min_ticks": 0},
        {"departure_delay": -1},
        {"departure_delay": 1.5},
        {"robot_budget": -1},
        {"survivor_budget": np.nan},
        {"dt": 0},
        {"dt": np.inf},
        {"max_dp_cells": 1},
        {"max_numeric_bytes": 1},
    ],
)
def test_invalid_or_truncated_models_raise_clear_errors(updates):
    with pytest.raises(ValueError):
        certify_rescue(np.ones((1, 3, 3)), (0, 1), (1, 2), **updates)


@pytest.mark.parametrize(
    "hazards",
    [
        np.ones((0, 3, 3)),
        np.ones((1, 3)),
        np.full((1, 3, 3), np.nan),
        np.full((1, 3, 3), -1),
        np.full((1, 3, 3), np.inf),
    ],
)
def test_invalid_scenario_arrays_are_rejected(hazards):
    with pytest.raises(ValueError):
        certify_rescue(hazards, (0, 1), (1, 2))


@pytest.mark.parametrize(
    "approach,escort",
    [
        ((), (1, 2)),
        ((0, 1), ()),
        ((0, 1), (0, 2)),
        ((-1, 1), (1, 2)),
        ((0, 3), (3, 2)),
        ((0.0, 1), (1, 2)),
    ],
)
def test_invalid_paths_are_rejected(approach, escort):
    with pytest.raises(ValueError):
        certify_rescue(np.ones((1, 4, 3)), approach, escort)


def test_floating_point_overflow_is_rejected_instead_of_certified():
    with pytest.raises(ValueError, match="float64"):
        certify_rescue(np.full((1, 4, 3), 1e308), (0, 1), (1, 2), escort_min_ticks=2)


def test_integer_kernel_exactly_matches_python_integer_exhaustive_enumeration():
    for flat in itertools.product((0, 1), repeat=8):
        hazards = np.array(flat, dtype=np.int64).reshape(1, 4, 2)
        result = certify_integer_rescue(
            hazards,
            (0,),
            (0, 1, 0),
            escort_min_ticks=(1, 1),
            escort_max_ticks=(2, 2),
            robot_budget=4,
            survivor_budget=4,
        )
        expected = 0
        for first, second in itertools.product((1, 2), repeat=2):
            nodes = [0] * first + [1] * second
            expected = max(
                expected,
                sum(int(hazards[0, tick, node]) for tick, node in enumerate(nodes)),
            )
        assert result.robot_bound == result.survivor_bound == expected
        assert (
            result.robot_witness.exposure
            == result.survivor_witness.exposure
            == expected
        )
        assert result.feasible == (expected <= 4)


def test_integer_arithmetic_boundary_and_exact_budget_comparison():
    half = 2**51
    hazards = np.array([[[half, 0], [0, half]]], dtype=np.int64)
    result = certify_integer_rescue(hazards, (0, 1), (1, 0), robot_budget=2**52 - 1)
    assert result.robot_bound == 2**52
    assert result.survivor_bound == half
    assert not result.feasible
    assert certify_integer_rescue(hazards, (0, 1), (1, 0), robot_budget=2**200).feasible
    empty = certify_integer_rescue(
        np.zeros((1, 0, 1), dtype=np.int64),
        (0,),
        (0,),
        robot_budget=0,
        survivor_budget=0,
    )
    assert empty.robot_bound == empty.survivor_bound == 0


@pytest.mark.parametrize(
    "rates",
    [
        np.ones((1, 2, 2), dtype=float),
        np.ones((1, 2, 2), dtype=bool),
        np.ones((1, 2, 2), dtype=object),
        np.full((1, 2, 2), -1, dtype=np.int64),
        np.full((1, 2, 2), 2**51 + 1, dtype=np.int64),
        np.full((1, 2, 2), 2**63, dtype=np.uint64),
    ],
)
def test_integer_wrapper_rejects_inexact_or_out_of_range_rates(rates):
    with pytest.raises(ValueError):
        certify_integer_rescue(rates, (0, 1), (1, 0))


@pytest.mark.parametrize(
    "options",
    [
        {"dt": 0.5},
        {"dt": True},
        {"robot_budget": 1.0},
        {"survivor_budget": -1},
        {"robot_budget": np.nan},
        {"robot_budget": -np.inf},
        {"survivor_budget": True},
    ],
)
def test_integer_wrapper_rejects_fractional_or_invalid_budget_units(options):
    with pytest.raises(ValueError):
        certify_integer_rescue(
            np.ones((1, 2, 2), dtype=np.int64), (0, 1), (1, 0), **options
        )
