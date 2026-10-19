import json
from pathlib import Path

import numpy as np
import pytest

from eval.lag_rescue_validation import exhaustive_replay, graph_oracle, replay_mission


def test_installed_cli_does_not_require_unshipped_test_sources(tmp_path, monkeypatch):
    import ember.rescue_certificate as kernel
    import eval.lag_rescue_validation as validation
    from swarmrescue.cli import main

    root = tmp_path / "site-packages"
    for name, source in (
        ("eval/lag_rescue_validation.py", validation.__file__),
        ("ember/rescue_certificate.py", kernel.__file__),
    ):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(source).read_bytes())
    monkeypatch.setattr(
        validation, "__file__", str(root / "eval/lag_rescue_validation.py")
    )
    output = tmp_path / "evidence"
    main(["certificate-evaluation", "--missions", "1", "--output-dir", str(output)])
    protocol = json.loads((output / "protocol.json").read_text())
    assert not protocol["tests_bundled_with_runtime"]
    assert protocol["test_source_sha256"] is None
    assert len(protocol["source_sha256"]) == 2


def test_external_graph_matches_exhaustive_complete_missions():
    rng = np.random.default_rng(411)
    for _ in range(12):
        field = rng.uniform(0, 2, (3, 14, 5))
        options = dict(
            approach_ticks=(1, 2),
            departure_delay=1,
            escort_min_ticks=(1, 2),
            escort_max_ticks=(3, 3),
            dt=0.3,
        )
        graph = graph_oracle(field, (0, 1, 2), (2, 3, 4), **options)
        direct, count = exhaustive_replay(field, (0, 1, 2), (2, 3, 4), **options)
        assert count == 18
        assert np.allclose([graph.robot_bound, graph.survivor_bound], direct)


def test_longest_duration_can_miss_worst_exposure():
    field = np.zeros((1, 5, 3))
    field[0, 1, 1] = 10
    graph = graph_oracle(
        field, (0,), (0, 1, 2), escort_min_ticks=(1, 1), escort_max_ticks=(2, 1)
    )
    slow = replay_mission(field, (0,), (0, 1, 2), (2, 1))
    assert graph.survivor_bound == 10
    assert graph.survivor_durations == (1, 1)
    assert slow["doses"][0, 1] == 0


def test_scenario_is_fixed_across_waiting_and_escort():
    field = np.zeros((2, 4, 3))
    field[0, 0, 1] = 4  # Victim waiting during robot approach.
    field[1, 1, 1] = 7  # A different scenario during escort.
    graph = graph_oracle(field, (0, 1), (1, 2))
    reset = graph_oracle(field.max(axis=0, keepdims=True), (0, 1), (1, 2))
    assert graph.survivor_bound == 7
    assert reset.survivor_bound == 11


def test_horizon_cannot_drop_admissible_late_schedules():
    field = np.zeros((1, 3, 3))
    with pytest.raises(ValueError, match="every admissible"):
        graph_oracle(field, (0, 1), (1, 2), escort_min_ticks=1, escort_max_ticks=3)
    with pytest.raises(ValueError, match="every admissible"):
        exhaustive_replay(field, (0, 1), (1, 2), escort_min_ticks=1, escort_max_ticks=3)


def test_singleton_route_and_predeparture_wait_exposure():
    field = np.ones((2, 4, 2))
    field[:, :, 1] = 3
    graph = graph_oracle(field, (0, 1), (1,), departure_delay=2)
    assert graph.robot_bound == 3
    assert graph.survivor_bound == 9
    assert graph.latest_completion == 3
    empty = graph_oracle(np.zeros((1, 0, 1)), (0,), (0,))
    assert empty.robot_bound == empty.survivor_bound == 0
    assert empty.latest_completion == 0


def test_kernel_and_generic_external_method_agree():
    from ember.rescue_certificate import certify_rescue

    rng = np.random.default_rng(913)
    for _ in range(20):
        field = rng.uniform(0, 1, (3, 16, 5))
        options = dict(
            approach_ticks=(2, 1),
            departure_delay=1,
            escort_min_ticks=(1, 2),
            escort_max_ticks=(3, 4),
            dt=0.7,
        )
        graph = graph_oracle(field, (0, 1, 2), (2, 3, 4), **options)
        result = certify_rescue(field, (0, 1, 2), (2, 3, 4), **options)
        assert np.allclose(
            [result.robot_bound, result.survivor_bound],
            [graph.robot_bound, graph.survivor_bound],
        )


def test_integer_kernel_matches_arbitrary_precision_external_graph_exactly():
    from ember.rescue_certificate import certify_integer_rescue

    rng = np.random.default_rng(3913)
    for _ in range(20):
        field = rng.integers(0, 500, (3, 16, 5), dtype=np.int64)
        options = dict(
            approach_ticks=(2, 1),
            departure_delay=1,
            escort_min_ticks=(1, 2),
            escort_max_ticks=(3, 4),
            dt=1,
        )
        graph = graph_oracle(field, (0, 1, 2), (2, 3, 4), **options)
        result = certify_integer_rescue(field, (0, 1, 2), (2, 3, 4), **options)
        assert isinstance(graph.robot_bound, int)
        assert isinstance(graph.survivor_bound, int)
        assert result.robot_bound == graph.robot_bound
        assert result.survivor_bound == graph.survivor_bound
