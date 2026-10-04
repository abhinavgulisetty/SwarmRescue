"""Procedural world/noise evaluation contracts."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from eval.procedural import (
    PROCEDURAL_METHOD_ORDER,
    ProceduralEvaluationConfig,
    evaluate_procedural,
    noise_rng,
    write_procedural_csv,
    write_procedural_report,
)


def test_held_out_worlds_are_sealed_by_default() -> None:
    with pytest.raises(ValueError, match="held-out world"):
        ProceduralEvaluationConfig(
            world_seeds=(10_000,),
            noise_seeds=(0,),
        )

    config = ProceduralEvaluationConfig(
        world_seeds=(10_000,),
        noise_seeds=(0,),
        allow_held_out=True,
    )
    assert config.split == "held_out"


def test_noise_streams_are_stable_and_separated() -> None:
    first = noise_rng(7, 3, "sensor_noise").normal(size=8)
    replay = noise_rng(7, 3, "sensor_noise").normal(size=8)
    other_noise = noise_rng(7, 4, "sensor_noise").normal(size=8)
    other_stream = noise_rng(7, 3, "radio_loss").normal(size=8)

    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first, other_noise)
    assert not np.array_equal(first, other_stream)


def test_procedural_report_uses_world_units_and_dynamic_trajectories(tmp_path) -> None:
    config = ProceduralEvaluationConfig(
        world_seeds=(0, 1),
        noise_seeds=(0, 1),
        bootstrap_samples=50,
        hazard_horizon_s=60.0,
    )

    report = evaluate_procedural(config, root=tmp_path)

    assert report.schema_version == 10
    assert len(report.runs) == 4
    assert report.protocol["experimental_unit"] == ["world_seed", "noise_seed"]
    assert report.protocol["world_varies_by_seed"] is True
    assert report.protocol["voi_uses_survivor_truth"] is False
    assert report.protocol["ember_belief_state"] == "independent_per_robot_gbp"
    assert report.protocol["ember_routing_arithmetic"] == "integer_log_q16"
    assert report.protocol["ember_escort_policy"] == "deadline_lmdp"
    assert report.protocol["voi_source_weight"] == "ble_estimate_nearest_graph_node"
    assert report.protocol["radio_loss_eventwise_paired"] is True
    assert report.protocol["bootstrap_unit"] == "world_seed"
    assert report.protocol["physics_collapse_used"] is False
    assert report.protocol["local_search_after_estimate"] is True
    assert report.protocol["communication_policy_varies_by_method"] is True
    assert report.protocol["communication_efficiency_comparison_available"] is True
    assert report.protocol["lmdp_control_temperature"] == 0.25
    assert report.protocol["uncertainty_beta"] == 1.0
    assert report.protocol["survivor_waiting_exposure_included"] is True
    assert report.protocol["search_assignment"] == (
        "ble_estimate_exact_minimum_distance_matching"
    )
    assert (
        report.protocol["local_search_policy"]
        == "deterministic_blind_depth_first_graph_walk"
    )
    assert report.protocol["ble_localization_dimensions"] == 3
    assert report.protocol["failure_times_censored_at_horizon"] is True
    assert "distributed_b3_consensus" not in report.unavailable_metrics
    assert report.summary["claim_checks"]["b3_reference_evaluated"] is True
    assert report.summary["claim_checks"]["b3_distributed_consensus_evaluated"] is True
    assert report.summary["claim_checks"]["c3_voi_radio_evaluated"] is True
    assert report.summary["claim_checks"]["c3_mean_radio_bytes_lower"] is True
    assert report.summary["communication"]["mean_reduction_ratio"] > 0.0
    assert not report.summary["claim_checks"]["paper_superiority_claim_ready"]

    fingerprints: dict[int, set[str]] = {}
    for run in report.runs:
        fingerprints.setdefault(run.world_seed, set()).add(run.world_fingerprint)
        assert set(run.methods) == set(PROCEDURAL_METHOD_ORDER)
        for result in run.methods.values():
            assert len(result.trajectories) == config.survivor_count
            assert 0.0 <= result.contact_completion_rate <= 1.0
            assert (
                0.0 <= result.rescue_completion_rate <= result.contact_completion_rate
            )
            assert result.time_to_first_contact_s >= config.observation_time_s
            assert result.mean_contact_time_s >= result.time_to_first_contact_s
            assert result.mean_evacuation_time_s >= 0.0
            assert result.mean_dynamic_exposure_proxy >= 0.0
            assert result.mean_rescue_time_s >= result.mean_contact_time_s
            assert result.mean_clairvoyant_regret >= 0.0
            assert math.isfinite(result.mean_objective_cost)
            for trajectory in result.trajectories:
                assert trajectory.contact_time_s >= config.observation_time_s
                assert trajectory.rescue_time_s >= trajectory.contact_time_s

    assert all(len(items) == 1 for items in fingerprints.values())
    assert len({next(iter(items)) for items in fingerprints.values()}) == 2
    ember_dose = report.summary["methods"]["ember"]["mean_dynamic_exposure_proxy"]
    assert ember_dose["world_count"] == 2

    routes = {
        tuple(trajectory.route for trajectory in run.methods[method].trajectories)
        for run in report.runs
        if run.world_seed == 0 and run.noise_seed == 0
        for method in PROCEDURAL_METHOD_ORDER
    }
    assert len(routes) >= 2

    json_path = tmp_path / "procedural.json"
    csv_path = tmp_path / "procedural.csv"
    write_procedural_report(report, json_path)
    write_procedural_csv(report, csv_path)

    decoded = json.loads(json_path.read_text(encoding="utf-8"))
    assert decoded["protocol"]["world_seed_count"] == 2
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 * len(PROCEDURAL_METHOD_ORDER)


def test_voi_schedule_is_invariant_to_hidden_survivor_truth(monkeypatch) -> None:
    from dataclasses import replace

    import eval.procedural as procedural
    from eval.procedural import (
        _schedule_voi_packets,
        build_procedural_world,
        sample_exogenous_streams,
    )

    config = ProceduralEvaluationConfig(world_seeds=(0,), noise_seeds=(0,))
    world = build_procedural_world(0, config)
    streams = sample_exogenous_streams(world, 0)
    # Hold all estimator inputs fixed while changing evaluation-only truth.
    changed = replace(world, survivor_nodes=world.exit_nodes)
    sources = tuple(
        (packet.sender, packet)
        for packet in streams.received_packets
    )
    weights = []
    original_solve = procedural.solve_adjoint

    def capture_source(model, source):
        weights.append(source.copy())
        return original_solve(model, source)

    monkeypatch.setattr(procedural, "solve_adjoint", capture_source)
    results = [
        _schedule_voi_packets(
            candidate,
            sources,
            config.voi_packet_budget_bytes,
            config.lmdp_control_temperature,
            noise_rng(0, 0, "radio_loss"),
            streams.localizations,
        )
        for candidate in (world, changed)
    ]
    np.testing.assert_array_equal(weights[0], weights[1])
    assert np.sum(weights[0]) == len(streams.localizations)
    assert results[0][1].broadcasts > 0
    assert results[0][1] == results[1][1]
    assert [packet.encode() for packet in results[0][0]] == [
        packet.encode() for packet in results[1][0]
    ]


def test_voi_budget_does_not_remove_local_sensor_evidence() -> None:
    from eval.procedural import build_procedural_world, sample_exogenous_streams

    config = ProceduralEvaluationConfig(world_seeds=(0,), noise_seeds=(0,))
    world = build_procedural_world(0, config)
    streams = sample_exogenous_streams(world, 0, voi_packet_budget_bytes=20)
    for robot in range(config.robot_count):
        def local_cells(packets, sender=robot):
            return {
                int(cell) for packet in packets if packet.sender == sender
                for cell in packet.cells
            }

        assert local_cells(streams.voi_received_by_robot[robot]) == local_cells(
            streams.received_by_robot[robot]
        )
