"""Contracts for the Phase-2 C4 resilience grid."""

from __future__ import annotations

import pytest

from eval.resilience import ResilienceConfig, evaluate_resilience


def test_resilience_grid_recovers_without_failure_detector(tmp_path) -> None:
    config = ResilienceConfig(
        world_seeds=(0,),
        failure_levels=(0.0, 0.5, 1.0),
        outage_sweeps=4,
        robot_counts=(8,),
        max_recovery_sweeps=200,
    )

    report = evaluate_resilience(config, root=tmp_path)

    assert report.schema_version == 2
    assert len(report.runs) == 3
    assert report.protocol["failure_detector"] == "none"
    assert report.summary["claim_checks"]["all_partitions_recovered"] is True
    assert report.summary["claim_checks"]["uncertainty_grows_after_death"] is True
    assert report.summary["claim_checks"]["mission_integration_complete"] is False
    assert report.summary["claim_checks"]["paper_claim_ready"] is False
    assert report.runs[0].failure_level == 0.0
    assert report.runs[0].recovery_half_life_sweeps == 0
    assert report.runs[-1].death_variance_ratio > 1.0
    assert report.runs[-1].recovery_half_life_sweeps > 0


def test_resilience_grid_keeps_held_out_worlds_sealed() -> None:
    with pytest.raises(ValueError, match="held-out"):
        ResilienceConfig(world_seeds=(10_000,))
