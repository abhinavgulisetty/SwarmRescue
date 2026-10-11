"""A corrupted evidence witness must not produce a successful audit."""

from copy import deepcopy

import pytest

from eval.exhaustive_deadline import simple_path_optima
from eval.paper_evidence import holm_adjust, paired_analysis
from eval.routing_certificates import (
    CertificateError,
    bellman_table,
    verify_result,
    verify_table,
)


def fixture():
    model = {
        "adjacent": [[[1, 1, 7], [2, 2, 1]], [[3, 1, 7]], [[3, 2, 1]], []],
        "start": 0,
        "goals": [3],
        "deadline": 2,
    }
    result = {
        "route": [0, 1, 3],
        "travel_steps": 2,
        "cost_q16": 14,
        "feasible": True,
        "optimal": True,
        "lower_bound_q16": 14,
        "arena_high_water_bytes": 200,
    }
    return model, result


def test_certificate_matches_independent_path_enumeration():
    model, result = fixture()
    for deadline, expected in enumerate(
        simple_path_optima(model["adjacent"], 0, {3}, 5)
    ):
        model["deadline"] = deadline
        assert verify_table(model, bellman_table(model)) == expected
    model["deadline"] = 2
    verify_result(model, result, 14, arena_bytes=200)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cost_q16", 13),
        ("travel_steps", 1),
        ("lower_bound_q16", 15),
        ("route", [0, 2, 3]),
        ("route", [0, 3]),
        ("arena_high_water_bytes", 201),
    ],
)
def test_rejects_corrupted_route_claim(field, value):
    model, result = fixture()
    result[field] = value
    with pytest.raises(CertificateError):
        verify_result(model, result, 14, arena_bytes=200)


def test_rejects_false_infeasibility_and_accepts_unresolved():
    model, result = fixture()
    result.update(feasible=False, optimal=True)
    with pytest.raises(CertificateError, match="false infeasibility"):
        verify_result(model, result, 14)
    result["optimal"] = False
    verify_result(model, result, 14)


def test_rejects_a_suboptimal_route_declared_exact():
    model, result = fixture()
    model["deadline"] = 4
    result["lower_bound_q16"] = 2
    with pytest.raises(CertificateError, match="false optimality"):
        verify_result(model, result, 2)


def test_rejects_tampered_bellman_witness():
    model, _ = fixture()
    table = bellman_table(model)
    for budget, node, bad in [(2, 0, 13), (0, 0, 0), (0, 3, None)]:
        corrupted = deepcopy(table)
        corrupted[budget][node] = bad
        with pytest.raises(CertificateError, match="Bellman mismatch"):
            verify_table(model, corrupted)


def test_holm_adjustment_is_monotone_in_sorted_order():
    assert holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04}) == {
        "a": 0.03,
        "b": 0.06,
        "c": 0.06,
    }


def test_statistics_keep_seed_clusters_and_exclude_nonexact_pairs():
    rows = []
    for seed in range(3):
        for _ in range(4):
            rows.append(
                {
                    "seed": seed,
                    "optimum_q16": 1,
                    "results": {
                        "candidate": {"optimal": True, "median_ms": 1},
                        "baseline": {"optimal": True, "median_ms": 2},
                    },
                }
            )
    rows.append(
        {
            "seed": 3,
            "optimum_q16": 1,
            "results": {
                "candidate": {"optimal": False, "median_ms": 0.01},
                "baseline": {"optimal": True, "median_ms": 100},
            },
        }
    )
    report = paired_analysis(rows, "baseline", "candidate")
    assert report["independent_seed_blocks"] == 3
    assert report["exact_pairs"] == 12
    assert report["candidate_geomean_speedup"] == pytest.approx(2)
    # Three clusters, not twelve independent observations: two of eight flips.
    assert report["two_sided_block_sign_flip_p"] == 0.25
