"""Tests for the development-only Phase-2 component benchmark."""

from __future__ import annotations

from eval.phase2 import Phase2BenchmarkConfig, evaluate_phase2


def test_phase2_benchmark_keeps_claims_gated() -> None:
    report = evaluate_phase2(
        Phase2BenchmarkConfig(
            world_seeds=(0, 1),
            tolerance=1.0e-7,
        )
    )

    assert len(report.runs) == 2
    assert report.schema_version == 2
    assert report.protocol["split"] == "development"
    assert report.protocol["held_out_seeds_opened"] is False
    assert report.summary["nca"]["all_converged"] is True
    assert report.summary["nca"]["parameter_bytes"] == 2_465
    assert report.summary["voi"]["all_adjoint_converged"] is True
    assert report.summary["lambda_diagnostic"]["implemented"] is True
    assert "lambda_diagnostic" not in report.summary["voi"]
    assert report.summary["claim_checks"]["c3_component_gate"] is True
    assert report.summary["claim_checks"]["mission_integration_complete"] is False
    assert report.summary["claim_checks"]["paper_claim_ready"] is False
