"""Contracts for the hash-locked Phase 3 protocol."""

from __future__ import annotations

import json

import pytest

from eval.heldout import DEFAULT_PROTOCOL, load_frozen_protocol


def test_frozen_protocol_covers_complete_heldout_set() -> None:
    protocol, digest = load_frozen_protocol()

    assert len(digest) == 64
    assert protocol["status"] == "frozen_before_held_out"
    assert protocol["evaluation"]["world_seed_first"] == 10_000
    assert protocol["evaluation"]["world_seed_last"] == 10_049
    assert protocol["evaluation"]["lmdp_control_temperature"] == 0.25
    assert protocol["claim_gates"]["nca_speedup_minimum"] == 2.0


def test_frozen_protocol_rejects_post_freeze_changes(tmp_path) -> None:
    protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
    protocol["evaluation"]["lmdp_control_temperature"] = 1.0
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen_protocol(changed)


def test_heldout_replay_is_not_labeled_fresh_confirmatory_evidence(
    monkeypatch, tmp_path
) -> None:
    from eval.heldout import evaluate_frozen_heldout
    from eval.procedural import ProceduralEvaluationReport

    historical = ProceduralEvaluationReport(
        schema_version=10,
        protocol={},
        provenance={},
        unavailable_metrics=(),
        runs=(),
        summary={},
    )
    def legacy_replay(config, **kwargs):
        assert config.escort_policy == "lmdp"
        return historical

    monkeypatch.setattr("eval.heldout.evaluate_procedural", legacy_replay)
    replay = evaluate_frozen_heldout(root=tmp_path)

    assert replay.protocol["evaluation_phase"] == "historical_held_out_replay"
    assert replay.protocol["held_out_previously_opened"] is True
    assert replay.protocol["implementation_locked_by_protocol"] is False
    assert replay.protocol["fresh_confirmatory_evidence"] is False
