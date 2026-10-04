"""Contracts for the paired development ablation grid."""

from __future__ import annotations

import pytest

from eval.ablations import (
    ABLATION_VARIANTS,
    AblationConfig,
    evaluate_ablations,
)


def test_ablation_grid_runs_all_contrasts_and_keeps_claims_gated(tmp_path) -> None:
    report = evaluate_ablations(
        AblationConfig(world_seeds=(0, 1)),
        root=tmp_path,
    )

    assert report.schema_version == 2
    assert report.protocol["split"] == "development"
    assert report.protocol["control_temperature"] == 0.25
    assert len(report.runs) == 2 * len(ABLATION_VARIANTS)
    assert len(report.q16_runs) == 2
    assert len(report.log_q16_runs) == 2
    assert report.summary["log_q16"]["all_fixed_points_reached"] is True
    assert report.summary["log_q16"]["mean_route_completion_ratio"] == 1.0
    assert report.summary["log_q16"]["mean_route_agreement_ratio"] == 1.0
    assert set(report.summary["variants"]) == set(ABLATION_VARIANTS)
    assert all(run.solver_converged for run in report.runs)
    assert all(run.all_routes_reached_exit for run in report.runs)
    assert report.summary["q16"]["all_fixed_points_reached"] is True
    assert 0.0 <= report.summary["q16"]["mean_route_completion_ratio"] <= 1.0
    assert (
        report.summary["claim_checks"]["required_ablation_grid_complete"]
        is False
    )
    assert report.summary["claim_checks"]["held_out_seeds_opened"] is False
    assert report.summary["claim_checks"]["physical_mission_validated"] is False
    assert report.summary["claim_checks"]["paper_claim_ready"] is False


def test_ablation_grid_keeps_held_out_worlds_sealed() -> None:
    with pytest.raises(ValueError, match="held-out"):
        AblationConfig(world_seeds=(10_000,))
