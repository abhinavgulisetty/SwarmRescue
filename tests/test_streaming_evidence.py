"""Streaming likelihood replacement, replay safety, and aggregate aging."""

import numpy as np
import pytest

from ember.comms import packetize_gaussians
from ember.robot_state import RobotState
from tests.test_robot_state import _graph


def packet(source, sequence, mean, variance=0.25, cell=0):
    return packetize_gaussians(
        source, sequence, np.array([cell]), np.array([mean]), np.array([variance])
    )[0]


def test_ten_thousand_readings_keep_learning_in_one_slot():
    robot = RobotState(
        0, _graph(), np.zeros(3), factor_capacity=1, latest_evidence=True
    )
    for sequence in range(10000):
        robot.enqueue(packet(0, sequence, float(sequence % 7)))
        assert robot.consume() == 1
    assert len(robot.seen_factors) == len(robot.evidence_slots) == 1
    assert robot.replaced_factors == 9999 and robot.dropped_factors == 0
    assert robot.engine.local_precision[0] == 29
    assert robot.engine.local_information[0] == 7.5 + 4 * (9999 % 7)
    assert robot.engine.belief().variance[0] == pytest.approx(1 / 29)


def test_source_sequences_are_independent_and_old_readings_cannot_replay():
    robot = RobotState(0, _graph(), np.zeros(3), latest_evidence=True)
    for item in (packet(0, 10, 1), packet(1, 2, 3), packet(0, 11, 5)):
        robot.enqueue(item)
        robot.consume()
    assert robot.engine.local_precision[0] == 33
    assert robot.engine.local_information[0] == 7.5 + 4 * (5 + 3)
    before = robot.engine.local_information.copy()
    for item in (packet(0, 10, 100), packet(0, 11, 100), packet(1, 1, 100)):
        robot.enqueue(item)
    assert robot.consume() == 0 and robot.stale_factors == 3
    np.testing.assert_array_equal(before, robot.engine.local_information)


def test_replacement_after_aging_keeps_background_and_other_sources():
    robot = RobotState(0, _graph(), np.zeros(3), latest_evidence=True)
    robot.enqueue(packet(0, 0, 2))
    robot.enqueue(packet(1, 0, 4))
    robot.consume()
    ratio = 1 / (1 + 0.2 * 33)
    robot.age_evidence(np.array([0]), 0.2)
    robot.enqueue(packet(0, 1, 6, variance=0.5))
    robot.consume()
    assert robot.engine.local_precision[0] == pytest.approx(29 * ratio + 2)
    assert robot.engine.local_information[0] == pytest.approx((7.5 + 16) * ratio + 12)
    assert robot.engine.local_precision[0] > 0


def test_geometry_transfer_preserves_aged_contributions_and_replay_watermarks():
    old = RobotState(0, _graph(), np.zeros(3), latest_evidence=True)
    old.enqueue(packet(0, 0, 2, cell=0))
    old.enqueue(packet(1, 3, 4, cell=1))
    old.consume()
    old.age_evidence(np.array([0, 1]), 0.02)
    new = RobotState(0, _graph(), np.zeros(3), latest_evidence=True)
    mapping = {0: 2}
    new.engine.local_precision[2] = old.engine.local_precision[0]
    new.engine.local_information[2] = old.engine.local_information[0]
    new.seen_factors = {(0, 0, 2)}
    new.transfer_evidence(old, mapping)
    assert new.evidence_slots == {(0, 2): 0}
    new.enqueue(packet(0, 0, 99, cell=2))
    assert new.consume() == 0
    new.enqueue(packet(0, 1, 3, cell=2))
    assert new.consume() == 1
    ratio = 1 / (1 + 0.02 * 29)
    assert new.engine.local_precision[2] == pytest.approx(25 * ratio + 4)
    assert new.engine.local_information[2] == pytest.approx(7.5 * ratio + 12)


def test_full_table_still_replaces_known_source_cells():
    robot = RobotState(
        0,
        _graph(),
        np.zeros(3),
        factor_capacity=1,
        latest_evidence=True,
        drop_on_factor_overflow=True,
    )
    robot.enqueue(packet(0, 0, 2))
    robot.consume()
    robot.enqueue(packet(1, 0, 4))
    assert robot.consume() == 0
    robot.enqueue(packet(0, 1, 6))
    assert robot.consume() == 1
    assert robot.dropped_factors == 1 and len(robot.evidence_slots) == 1


def test_latest_storage_is_arena_backed_and_charged_to_numeric_peak():
    robot = RobotState(
        0, _graph(), np.zeros(3), factor_capacity=9, latest_evidence=True
    )
    storage = np.frombuffer(robot.arena._storage, dtype=np.uint8)
    assert np.shares_memory(storage, robot.evidence_values)
    assert np.shares_memory(storage, robot.evidence_sequences)
    base = RobotState(0, _graph(), np.zeros(3), factor_capacity=9)
    assert robot.arena.used - base.arena.used == 18 * 9
