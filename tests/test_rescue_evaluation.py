"""Acceptance independently checks coordinates, custody and archived sources."""

import copy
import hashlib
import io
import json
import tarfile

import pytest

from eval.rescue import check_rescue_report, summarize


def _report():
    return {
        "accepted": True,
        "physical_survivor_evacuation": True,
        "rescue_acceptance_gate_met": True,
        "live_firmware_routing": True,
        "provenance": {
            "source_kind": "installed_distribution",
            "source_fingerprint": "a" * 64,
        },
        "metrics": {
            "seed": 7,
            "n_robots": 3,
            "survivors_hidden": 3,
            "survivors_contacted": 3,
            "survivors_evacuated": 3,
            "survivor_escort_robot": [2, 0, 1],
            "evacuation_footprint_m": [0, 0, 10, 10],
            "survivor_positions_m": [[-1, 1, 0], [-1, 2, 0], [-1, 3, 0]],
            "robot_positions_m": [[-1, 1, 0.14], [-1, 2, 0.14], [-1, 3, 0.14]],
            "survivor_contact_time_s": [10, 11, 12],
            "survivor_evacuation_time_s": [30, 31, 32],
            "survivor_exposure_proxy": [1, 2, 3],
            "firmware_robots": 2,
            "firmware_requests": 6,
        },
    }


@pytest.mark.parametrize(
    "fault", ["indoors", "fallen", "duplicate_owner", "time", "firmware"]
)
def test_success_label_cannot_override_physical_failure(fault):
    report = _report()
    check_rescue_report(report)
    m = report["metrics"]
    if fault == "indoors":
        m["survivor_positions_m"][0][0] = 1
    elif fault == "fallen":
        m["survivor_positions_m"][0][2] = -2
    elif fault == "duplicate_owner":
        m["survivor_escort_robot"] = [0, 0, 1]
    elif fault == "time":
        m["survivor_evacuation_time_s"][0] = 5
    else:
        m["firmware_requests"] = 0
    with pytest.raises(ValueError):
        check_rescue_report(report)


def _write(tmp_path, report, *, changed=False):
    p = tmp_path / f"seed{report['metrics']['seed']}.json"
    p.write_text(json.dumps(report))
    payload = b"runtime"
    metadata = {
        "source_fingerprint": report["provenance"]["source_fingerprint"],
        "source_files": [
            {"path": "ember/model.py", "sha256": hashlib.sha256(payload).hexdigest()}
        ],
    }
    with tarfile.open(p.with_suffix(".source.tar.gz"), "w:gz") as archive:
        for name, content in [
            ("bundle.json", json.dumps(metadata).encode()),
            ("installed/ember/model.py", b"changed" if changed else payload),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return p


def test_archive_corruption_blocks_release_acceptance(tmp_path):
    valid = _write(tmp_path, _report())
    assert summarize([valid])["accepted"]
    corrupt = _write(tmp_path, _report(), changed=True)
    result = summarize([corrupt])
    assert not result["accepted"]
    assert "archived source hash differs" in result["runs"][0]["issues"]


def test_paired_release_cannot_mix_source_versions(tmp_path):
    a = _report()
    b = copy.deepcopy(a)
    b["metrics"]["seed"] = 42
    b["provenance"]["source_fingerprint"] = "b" * 64
    result = summarize([_write(tmp_path, a), _write(tmp_path, b)])
    assert not result["accepted"] and not result["same_runtime_sources"]
