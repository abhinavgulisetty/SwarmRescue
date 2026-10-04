"""Contracts for paired, claim-aware experiment evaluation."""

from __future__ import annotations

import csv
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

import ember.mission as mission_module
from ember.baselines import BaselineResult
from ember.localization import LocalizationEstimate
from eval import (
    EvaluationConfig,
    evaluate,
    parse_seed_spec,
    write_report,
    write_runs_csv,
)


def test_seed_spec_uses_inclusive_ranges() -> None:
    assert parse_seed_spec("0:2,7,9:10") == (0, 1, 2, 7, 9, 10)


@pytest.mark.parametrize("specification", ("", "3:1", "1:2:3"))
def test_seed_spec_rejects_invalid_input(specification: str) -> None:
    with pytest.raises(ValueError):
        parse_seed_spec(specification)


def test_held_out_seeds_are_sealed_by_default() -> None:
    with pytest.raises(ValueError, match="sealed"):
        EvaluationConfig(seeds=(10_000,))

    config = EvaluationConfig(seeds=(10_000,), allow_held_out=True)
    assert config.split == "held_out"


def test_real_paired_report_is_an_outcome_agnostic_smoke_test(tmp_path) -> None:
    config = EvaluationConfig(seeds=(0, 1), bootstrap_samples=100)
    report = evaluate(config, root=tmp_path)
    comparison = report.summary["paired"]["ember_minus_b2"]

    assert comparison["wins"] + comparison["ties"] + comparison["losses"] == 2
    assert 0.0 <= comparison["same_route_rate"] <= 1.0
    assert not report.summary["claim_checks"]["paper_superiority_claim_ready"]
    assert "iso_13571_survival" in report.unavailable_metrics
    assert report.protocol["communication_policy_varies_by_method"] is False
    assert report.protocol["communication_efficiency_comparison_available"] is False
    assert "method_specific_communication_efficiency" in report.unavailable_metrics

    for run in report.runs:
        assert run.receiver_information_gain_nats >= 0.0
        assert math.isfinite(run.nats_per_byte)
        assert all(method.clairvoyant_regret >= 0.0 for method in run.methods.values())

    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "runs.csv"
    write_report(report, json_path)
    write_runs_csv(report, csv_path)

    decoded = json.loads(json_path.read_text(encoding="utf-8"))
    assert decoded["protocol"]["paired_common_random_numbers"] is True
    assert decoded["protocol"]["localization_coverage_test"].startswith("mahalanobis")
    assert len(decoded["runs"]) == 2
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert {row["method"] for row in rows} == {
        "b1_shortest",
        "b2_flooding_dijkstra",
        "ember",
    }


def _synthetic_mission(seed: int) -> SimpleNamespace:
    oracle = BaselineResult("oracle", (0,), 1.0, 0.0, 0)
    localization = LocalizationEstimate(
        np.zeros(2),
        np.eye(2),
        residual_rms_m=0.0,
        measurements=3,
    )
    return SimpleNamespace(
        succeeded=True,
        centralized_baseline=oracle,
        flooding_baseline=BaselineResult("b2", (0, 1), 1.0, 2.0, 100),
        ember_result=BaselineResult(
            "ember",
            (0, seed + 2),
            1.0,
            float(seed + 1),
            50,
        ),
        clairvoyant_baseline=oracle,
        localization_error_m=0.0,
        localization=localization,
        survivor_position_m=np.zeros(2),
        receiver_information_gain_nats=1.0,
        radio_bytes=100,
        radio_stats=SimpleNamespace(delivered=1),
        gbp_sweeps=7,
        routing_solution=SimpleNamespace(sweeps=5),
    )


def test_paired_summary_handles_ember_win_tie_and_loss(tmp_path) -> None:
    config = EvaluationConfig(seeds=(0, 1, 2), bootstrap_samples=100)
    report = evaluate(config, mission_runner=_synthetic_mission, root=tmp_path)

    comparison = report.summary["paired"]["ember_minus_b2"]
    assert comparison["wins"] == 1
    assert comparison["ties"] == 1
    assert comparison["losses"] == 1


def test_b2_policy_map_is_independent_of_ember_gbp(monkeypatch) -> None:
    monkeypatch.setattr(
        mission_module,
        "last_received_mean_map",
        lambda n_nodes, _packets, *, prior_mean: np.zeros(n_nodes),
    )

    result = mission_module.run_mission(0)

    assert result.flooding_baseline.route == result.centralized_baseline.route
    assert result.ember_result.route != result.flooding_baseline.route
    assert result.gbp_sweeps > result.gbp_result.sweeps
