"""Focused cases must demonstrate actual causes and physical outcomes."""

import json

import numpy as np
import pytest

from swarmrescue.cases import CASES, run_case, write_gallery


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    out = tmp_path_factory.mktemp("cases")
    return {name: run_case(name, out, render=False) for name in CASES}


def test_contact_requires_a_visible_local_reading_and_physical_approach(reports):
    report = reports["contact"]
    events = {e["event"]: e for e in report["events"]}
    assert report["accepted"] and report["occluded_samples"] > 0
    assert 0 < events["survivor_detected"]["time_s"] < events["local_contact"]["time_s"]
    assert events["local_contact"]["distance_m"] <= 0.7
    assert report["robot_travel_m"] > 2
    assert report["survivor_travel_m"] < 0.05
    assert "survivor_evacuated" not in events


def test_fallen_geometry_invalidates_the_direct_route_and_robot_drives_detour(reports):
    report = reports["reroute"]
    events = {e["event"]: e for e in report["events"]}
    assert report["accepted"]
    assert events["debris_released"]["time_s"] < events["route_replaced"]["time_s"]
    assert events["geometry_scanned"]["debris_position_m"][2] < 0.36
    original, replacement = map(
        np.array, (report["original_route_m"], report["final_route_m"])
    )
    assert original[:, 1].max() < 3
    assert replacement[:, 1].max() > 4
    assert abs(report["robot_final_m"][0] - 6.25) < 0.4
    assert report["robot_travel_m"] > 5


def test_escort_waits_and_moves_the_survivor_beyond_the_physical_boundary(reports):
    report = reports["escort"]
    events = {e["event"]: e for e in report["events"]}
    assert report["accepted"]
    assert events["local_contact"]["time_s"] < events["escort_waits"]["time_s"]
    assert events["escort_waits"]["time_s"] < events["survivor_evacuated"]["time_s"]
    assert report["survivor_final_m"][0] <= -0.5
    assert abs(report["survivor_final_m"][2]) < 0.25
    assert report["survivor_travel_m"] > 4
    assert report["robot_final_m"][0] < -0.5


def test_no_exit_stops_before_obstruction_without_a_fabricated_route(reports):
    report = reports["no-exit"]
    assert report["accepted"] and report["outcome"] == "safe_stop"
    assert report["final_route_m"] == []
    assert report["robot_final_m"][0] < 3
    after = [s["robot_m"] for s in report["samples"] if s["time_s"] >= 5]
    assert np.linalg.norm(np.array(after[-1]) - after[0]) < 0.02
    assert "destination_reached" not in {e["event"] for e in report["events"]}


def test_gallery_uses_measured_chapters_and_relative_portable_assets(tmp_path, reports):
    write_gallery(tmp_path, list(reports.values()))
    page = (tmp_path / "index.html").read_text()
    assert 'data-case="3"' in page
    assert "event.time_s / r.playback_speed" in page
    assert "http://" not in page and "https://" not in page
    assert json.loads((tmp_path / "summary.json").read_text())["accepted"]
