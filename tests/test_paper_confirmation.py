"""Replication uses every status and retains process/seed dependence."""

import pytest

from eval.paper_confirmation import all_exact, estimate


@pytest.mark.parametrize(
    "bad",
    [
        {"feasible": False, "optimal": True},
        {"feasible": True, "optimal": False},
        {"feasible": False, "optimal": False},
    ],
)
def test_nonexact_repeat_disqualifies_the_pair(bad):
    good = {"feasible": True, "optimal": True}
    assert all_exact({"samples": [good] * 7})
    assert not all_exact({"samples": [good, bad, good]})


def test_hierarchical_estimator_preserves_constant_process_ratio():
    pairs = []
    for seed in range(4):
        for capacity in (32, 128):
            rows = [
                {
                    "results": {
                        "native_exact": {"median_ms": p + 1},
                        "baseline": {"median_ms": 2 * (p + 1)},
                    }
                }
                for p in range(3)
            ]
            pairs.append(((seed, "synthetic", capacity), rows))
    result = estimate(pairs, "baseline", draws=1000)
    assert result["exact_pairs"] == 8
    assert result["informative_seed_blocks"] == 4
    assert result["candidate_geomean_speedup"] == pytest.approx(2)
    assert result["process_geomean_speedups"] == pytest.approx([2, 2, 2])
    assert result["process_and_seed_bootstrap_95ci"] == pytest.approx([2, 2])
