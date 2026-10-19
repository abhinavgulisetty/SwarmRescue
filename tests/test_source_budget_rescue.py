"""Independent forward dynamics and adversarial witnesses for the research model."""

import itertools

import numpy as np
import pytest

from eval.source_budget_rescue import (
    distinguish_coefficients,
    dose_certificate,
    exhaustive_checks,
    forward_vertex_oracle,
    packed_summary,
    quantized_upper,
    random_study,
    rescue_schedule,
    sharp_witnesses,
)


def test_complete_schedule_includes_waiting_and_slower_escort():
    schedule = rescue_schedule((0, 1, 2), (2, 1, 0), 3, delay=2, escort_ticks=3)
    assert schedule.pickup_tick == 4
    assert schedule.completion_tick == 10
    np.testing.assert_array_equal(
        schedule.occupancy.argmax(axis=2),
        [
            [0, 0, 0, 1, 2, 2, 2, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 1, 1, 1],
        ],
    )
    witnesses = sharp_witnesses()
    assert witnesses["waiting_exposure"]["survivor_dose"] == 6
    assert witnesses["scalar_noncomposability"]["complete_maxima"] == [2, 1]
    assert witnesses["snapshot_unsafe"]["worst_case"] == 7


def test_coefficients_match_forward_vertex_oracle_with_time_varying_base():
    transition = np.array([[0.8, 0.3], [0.2, 1.05]])
    injection = np.array([[1.0, 0.1, 0.0], [0.0, 0.2, 0.9]])
    occupancy = rescue_schedule((0, 1), (1, 0), 2, delay=2).occupancy
    base = np.arange(occupancy.shape[1] * 2).reshape(-1, 2) * 0.1
    result = dose_certificate(transition, injection, occupancy, 2.5, dt=0.25, base=base)
    oracle = forward_vertex_oracle(
        transition, injection, occupancy, 2.5, dt=0.25, base=base
    )
    np.testing.assert_allclose(result.worst_case, oracle, rtol=1e-13, atol=1e-13)
    assert np.all(result.pointwise_reset >= result.worst_case)
    zero = dose_certificate(transition, injection, occupancy, 0, dt=0.25, base=base)
    np.testing.assert_array_equal(zero.worst_case, zero.base_dose)


def test_pointwise_reset_is_sharply_overconservative():
    # Source one reaches the observed node only at tick zero, source two at tick one.
    transition = np.array([[0.0, 1.0], [0.0, 0.0]])
    occupancy = np.zeros((1, 2, 2))
    occupancy[:, :, 0] = 1
    result = dose_certificate(transition, np.eye(2), occupancy, 1)
    np.testing.assert_array_equal(result.coefficients, [[1, 1]])
    np.testing.assert_array_equal(result.worst_case, [1])
    np.testing.assert_array_equal(result.pointwise_reset, [2])


def test_scalar_summary_cannot_determine_future_feasibility():
    q, other, future = np.array([1, 0]), np.array([0, 1]), np.array([1, 0])
    assert max(q) == max(other)
    assert max(q + future) > 1 >= max(other + future)


def test_all_small_integer_vectors_are_distinguishable_and_pack_optimally():
    for bound, dimension in ((0, 3), (1, 4), (2, 3)):
        domain = list(itertools.product(range(bound + 1), repeat=dimension))
        packed = [packed_summary(q, bound) for q in domain]
        assert len({value for value, _ in packed}) == len(domain)
        bits = (len(domain) - 1).bit_length()
        assert all(count == bits and value < 2**bits for value, count in packed)
        for left, right in itertools.combinations(domain, 2):
            r = distinguish_coefficients(left, right, bound)
            assert (np.max(left + r) <= bound) != (np.max(right + r) <= bound)
    exhaustive = exhaustive_checks()
    assert exhaustive["coefficient_tick_cases"] == 729
    assert exhaustive["support_max_abs_error"] == 0
    assert exhaustive["pointwise_dominance_violations"] == 0
    assert exhaustive["integer_domain"]["distinction_violations"] == 0


def test_upward_quantization_preserves_safety_with_stated_error():
    rng = np.random.default_rng(8)
    for _ in range(100):
        q = rng.uniform(0, 10, (3, 4))
        base = rng.uniform(0, 1, 3)
        budget, delta, base_delta = 1.7, 0.05, 0.01
        exact = base + budget * q.max(axis=1)
        upper = quantized_upper(base, q, budget, delta, base_delta=base_delta)
        assert np.all(upper >= exact - 1e-12)
        assert np.all(upper - exact <= budget * delta + base_delta + 1e-12)
        limits = exact + 0.03
        assert np.all((upper <= limits) <= (exact <= limits))
    zero_budget = quantized_upper(
        np.array([0.12]), np.array([[9.0, 80.0]]), 0, 0.05, base_delta=0.1
    )
    np.testing.assert_allclose(zero_budget, [0.2])
    assert 0 <= zero_budget[0] - 0.12 <= 0.1


def test_waiting_feasibility_can_be_nonmonotone_even_in_a_positive_system():
    transition, injection = np.zeros((6, 6)), np.zeros((6, 1))
    transition[1, 2] = transition[2, 3] = transition[3, 4] = 1
    injection[4, 0] = 1
    doses = []
    for delay in (0, 1, 2):
        schedule = rescue_schedule((5, 0), (0, 1, 5), 6, delay=delay, escort_ticks=1)
        oracle = forward_vertex_oracle(transition, injection, schedule.occupancy, 1)
        doses.append(float(oracle[1]))
    assert doses == [0, 1, 0]
    assert [dose <= 0.5 for dose in doses] == [True, False, True]
    assert sharp_witnesses()["nonmonotone_delay"]["survivor_doses"] == doses


def test_paired_study_is_deterministic_and_certificates_have_no_unsafe_accepts():
    result = random_study(seed=17, systems=12, schedules_per_system=6)
    assert result == random_study(seed=17, systems=12, schedules_per_system=6)
    assert result["schedules"] == 72
    assert result["max_abs_error"] < 1e-11
    assert result["pointwise_dominance_violations"] == 0
    assert result["quantization_bound_violations"] == 0
    for method in ("fixed_source", "pointwise_reset", "quantized_upper"):
        assert result["rejection_tradeoff"][method]["unsafe_accepts"] == 0
    assert result["rejection_tradeoff"]["current_snapshot"]["unsafe_accepts"] > 0


@pytest.mark.parametrize("mutation", ["negative", "nonfinite", "occupancy", "budget"])
def test_invalid_models_are_rejected(mutation):
    transition, injection, occupancy, budget = (
        np.eye(2),
        np.eye(2),
        np.ones((1, 2, 2)) / 2,
        1,
    )
    if mutation == "negative":
        transition[0, 1] = -0.1
    elif mutation == "nonfinite":
        injection[0, 0] = np.nan
    elif mutation == "occupancy":
        occupancy *= 2
    else:
        budget = -1
    with pytest.raises(ValueError):
        dose_certificate(transition, injection, occupancy, budget)


def test_disconnected_schedule_and_equal_vectors_are_rejected():
    with pytest.raises(ValueError):
        rescue_schedule((0, 1), (0, 1), 2)
    with pytest.raises(ValueError):
        distinguish_coefficients((1, 0), (1, 0), 1)
