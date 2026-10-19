"""Complete missions, private cost rows and infeasible exits affect assignment."""

import itertools

import numpy as np
import pytest

from ember.rescue_planning import complete_rescue_assignment


def _line():
    positions = np.column_stack((np.arange(5), np.zeros((5, 2))))
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    return positions, edges


def test_selected_bids_include_waiting_escort_and_reachable_exit():
    positions, edges = _line()
    result = complete_rescue_assignment(
        positions,
        edges,
        np.array([0, 4]),
        np.array([[1, 3], [1, 3]]),
        np.array([0, 4]),
        np.ones((2, 5)),
        available_time_s=60,
    )
    assert result.robot_to_survivor == (0, 1)
    for bid in result.selected:
        assert bid.feasible
        assert bid.approach[-1] == bid.escort[0]
        assert bid.escort[-1] in (0, 4)
        assert bid.survivor_bound > 0
        assert bid.latest_completion_s <= 60


def test_contact_without_exit_cannot_become_a_feasible_bid():
    positions, edges = _line()
    result = complete_rescue_assignment(
        positions,
        edges[:2],
        np.array([0]),
        np.array([[2]]),
        np.array([4]),
        np.ones((1, 5)),
        available_time_s=60,
    )
    assert result.robot_to_survivor == (-1,)
    assert result.selected == (None,)
    assert result.rejected_pairs == 1


def test_unequal_team_sizes_preserve_exclusive_ownership():
    positions, edges = _line()
    result = complete_rescue_assignment(
        positions,
        edges,
        np.array([0, 0, 4]),
        np.array([[2], [2], [2]]),
        np.array([0, 4]),
        np.ones((3, 5)),
        available_time_s=60,
    )
    assert sorted(result.robot_to_survivor) == [-1, -1, 0]


def test_matching_equals_exhaustive_optimum_over_the_same_full_rescue_bids():
    positions, edges = _line()
    risk = np.array([[1, 2, 1, 4, 1], [1, 6, 1, 2, 1], [1, 1, 1, 1, 1]])
    result = complete_rescue_assignment(
        positions,
        edges,
        np.array([0, 4, 0]),
        np.array([[1, 2, 3]] * 3),
        np.array([0, 4]),
        risk,
        available_time_s=90,
    )
    values = {}
    for bid in result.bids:
        if bid.feasible:
            key = bid.robot, bid.survivor
            values[key] = min(values.get(key, np.inf), bid.objective)
    optimum = min(
        sum(values[r, v] for r, v in enumerate(permutation))
        for permutation in itertools.permutations(range(3))
    )
    assert sum(b.objective for b in result.selected) == pytest.approx(optimum)


def test_deadline_and_survivor_budget_can_withdraw_all_bids():
    positions, edges = _line()
    for options in ({"available_time_s": 0.5}, {"survivor_budget": 0.0}):
        arguments = {"available_time_s": 60, **options}
        result = complete_rescue_assignment(
            positions,
            edges,
            np.array([0]),
            np.array([[3]]),
            np.array([0]),
            np.ones((1, 5)),
            **arguments,
        )
        assert result.robot_to_survivor == (-1,)


def test_invalid_forecasts_are_rejected_before_planning():
    positions, edges = _line()
    with pytest.raises(ValueError, match="risk"):
        complete_rescue_assignment(
            positions,
            edges,
            np.array([0]),
            np.array([[3]]),
            np.array([0]),
            np.full((1, 5), np.nan),
            available_time_s=60,
        )


def _detour(deadline, **options):
    # The direct route crosses node 4. Risk weighting prefers a 12 m detour
    # through nodes 2 and 3 over the direct 2 m route in both directions.
    positions = np.array([[0, 0, 0], [2, 0, 0], [0, 5, 0], [2, 5, 0], [1, 0, 0]])
    edges = np.array([[0, 4], [4, 1], [0, 2], [2, 3], [3, 1]])
    return complete_rescue_assignment(
        positions,
        edges,
        np.array([0]),
        np.array([[1]]),
        np.array([0]),
        np.array([[0.1, 0.1, 0.01, 0.01, 50.0]]),
        available_time_s=deadline,
        approach_speed_m_s=1,
        escort_speed_m_s=1,
        dt_s=1,
        lag_fraction=0,
        **options,
    )


def test_short_higher_risk_route_retains_pair_when_low_risk_detour_misses_deadline():
    result = _detour(6)
    assert result.robot_to_survivor == (0,)
    assert result.rejected_pairs == 0
    bid = result.selected[0]
    assert bid.feasible
    assert bid.approach == (0, 4, 1)
    assert bid.escort == (1, 4, 0)
    assert bid.latest_completion_s == 4
    assert bid.robot_bound == pytest.approx(100.2)
    assert bid.survivor_bound == pytest.approx(50.3)


def test_pair_uses_cheapest_feasible_alternative_and_preserves_all_unique_bids():
    result = _detour(30)
    alternatives = [bid for bid in result.bids if bid.feasible]
    assert len(alternatives) == 4
    assert len({(bid.approach, bid.escort) for bid in alternatives}) == 4
    assert result.selected[0].objective == min(bid.objective for bid in alternatives)
    assert result.selected[0] == alternatives[0]
    assert alternatives[-1].objective > alternatives[0].objective
    rejected = _detour(30, survivor_budget=0)
    assert len(rejected.bids) == 4
    assert rejected.robot_to_survivor == (-1,)
    assert rejected.rejected_pairs == 1
