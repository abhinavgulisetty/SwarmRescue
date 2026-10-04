"""Contracts for the analytical MCU resource profile."""

from __future__ import annotations

import pytest

from eval.mcu_profile import MCUProfileConfig, evaluate_mcu_profile


def test_mcu_profile_reports_gated_analytical_resources(tmp_path) -> None:
    report = evaluate_mcu_profile(
        MCUProfileConfig(world_seeds=(0, 1)),
        root=tmp_path,
    )

    assert report.schema_version == 1
    assert len(report.runs) == 2
    assert (
        report.protocol["measurement_kind"]
        == "tier_a_analytical_not_emulator_measured"
    )
    assert report.summary["claim_checks"]["ram_below_200_kib"] is True
    assert (
        report.summary["claim_checks"]["deadline_miss_rate_below_5_percent"]
        is True
    )
    assert report.summary["claim_checks"]["emulator_calibration_complete"] is False
    assert report.summary["claim_checks"]["tier_b_complete"] is False
    assert all(run.arena_high_water_bytes > 2_465 for run in report.runs)
    assert all(0.0 < run.cycle_utilization < 1.0 for run in report.runs)


def test_mcu_profile_keeps_held_out_worlds_sealed() -> None:
    with pytest.raises(ValueError, match="held-out"):
        MCUProfileConfig(world_seeds=(10_000,))
