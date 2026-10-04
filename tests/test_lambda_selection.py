"""Contracts for development-only lambda selection."""

from __future__ import annotations

import pytest

from eval.lambda_selection import (
    LambdaSelectionConfig,
    evaluate_lambda_selection,
)


def test_lambda_selection_uses_development_regret_and_keeps_gate_closed(
    tmp_path,
) -> None:
    config = LambdaSelectionConfig(
        world_seeds=(0, 1),
        candidates=(0.5, 1.0),
    )

    report = evaluate_lambda_selection(config, root=tmp_path)

    assert report.schema_version == 1
    assert len(report.runs) == 4
    assert report.summary["selected_lambda"] in config.candidates
    assert report.summary["claim_checks"]["all_candidates_converged"] is True
    assert (
        report.summary["claim_checks"]["development_selection_complete"]
        is False
    )
    assert report.summary["claim_checks"]["held_out_seeds_opened"] is False
    assert report.summary["claim_checks"]["paper_claim_ready"] is False
    assert all(run.mean_static_regret >= 0.0 for run in report.runs)
    assert all(run.mean_optimism_gap >= 0.0 for run in report.runs)


def test_lambda_selection_keeps_held_out_worlds_sealed() -> None:
    with pytest.raises(ValueError, match="held-out"):
        LambdaSelectionConfig(world_seeds=(10_000,))
