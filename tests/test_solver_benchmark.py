"""The comparison uses reproducible inputs and includes all graph families."""

import numpy as np
import pytest

from eval import solver_benchmark
from eval.solver_benchmark import (
    DEVELOPMENT_SEEDS,
    EVALUATION_SEEDS,
    FAMILIES,
    make_model,
)
from mcu.native_log_lmdp import STRATEGIES, solve_native_log_q16


@pytest.mark.parametrize("family", FAMILIES)
def test_benchmark_models_reproduce_and_solve_identically(family):
    model = make_model(DEVELOPMENT_SEEDS[0], family)
    repeated = make_model(DEVELOPMENT_SEEDS[0], family)
    assert set(DEVELOPMENT_SEEDS).isdisjoint(EVALUATION_SEEDS)
    for name in (
        "row_ptr",
        "col_idx",
        "passive_prob",
        "state_cost",
        "exit_mask",
        "death_mask",
    ):
        np.testing.assert_array_equal(getattr(model, name), getattr(repeated, name))
    reference = solve_native_log_q16(model)
    assert reference.fixed_point
    for strategy in STRATEGIES:
        result = solve_native_log_q16(model, strategy=strategy)
        if not result.fixed_point:
            # The exact residual queue can exhaust its budget on sticky rows.
            # Benchmark reports retain this failure instead of calling it fast.
            assert family == "sticky" and strategy in (
                "prioritized",
                "bound_prioritized",
            )
            assert result.sweeps == model.config.max_sweeps
            continue
        assert result.fixed_point
        assert result.arena_high_water_bytes < 200 * 1024
        np.testing.assert_array_equal(
            result.negative_log_desirability, reference.negative_log_desirability
        )


def test_frozen_evaluation_rejects_modified_code_before_generating_models(
    monkeypatch, tmp_path
):
    kernel = solver_benchmark.Path(solver_benchmark.__file__).resolve().parents[1]
    original_source = (kernel / "firmware/log_lmdp.c").read_bytes()
    original_hash = solver_benchmark.hashlib.sha256

    def modified_hash(data=b""):
        return original_hash(data + b"modified" if data == original_source else data)

    def forbidden_generation(*args):
        pytest.fail("evaluation models must stay unopened after a source mismatch")

    monkeypatch.setattr(solver_benchmark.hashlib, "sha256", modified_hash)
    monkeypatch.setattr(solver_benchmark, "make_model", forbidden_generation)
    output = tmp_path / "evaluation.json"
    with pytest.raises(RuntimeError, match="differ from frozen evaluation"):
        solver_benchmark.main(["--split", "evaluation", "--output", str(output)])
    assert not output.exists()
