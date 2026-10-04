"""Paired algorithm-selection report and metric accounting."""

import pytest

from eval.innovation import LEGACY, evaluate_innovations
from eval.procedural import ProceduralEvaluationConfig


def test_comparison_reports_raw_dose_completion_and_common_worlds(tmp_path):
    report = evaluate_innovations(
        ProceduralEvaluationConfig(
            world_seeds=(0,),
            noise_seeds=(0,),
            bootstrap_samples=50,
        ),
        algorithms=(LEGACY, "adjoint_kl/canonical/deadline_pareto"),
        root=tmp_path,
    )
    assert len(report["runs"]) == 1
    assert set(report["ranking"]) == {LEGACY, "adjoint_kl/canonical/deadline_pareto"}
    assert report["protocol"]["source_packet_budgets_matched"] is True
    assert report["protocol"]["air_bytes_exactly_matched"] is False
    assert report["protocol"]["future_hazard_truth_available_to_policy"] is False
    for method in report["runs"][0]["methods"].values():
        assert (
            method["mean_unpenalized_exposure_proxy"]
            <= method["mean_dynamic_exposure_proxy"]
        )
    assert "b2_flooding_dijkstra" in report["summary"][LEGACY]["paired"]


def test_innovation_rejects_even_mixed_historical_heldout_inputs():
    config = ProceduralEvaluationConfig(
        world_seeds=(0, 10000),
        noise_seeds=(0,),
        allow_held_out=True,
    )
    with pytest.raises(ValueError, match="held-out"):
        evaluate_innovations(config)
