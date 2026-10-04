"""Map identity on the wire and safe coordinated graph transitions."""

import struct
from dataclasses import replace

import numpy as np
import pytest

from ember.belief import GaussianBelief
from ember.comms import (
    MAX_MAP_EPOCH,
    MAX_PAYLOAD_BYTES,
    MAX_UPDATES,
    GaussianPacket,
    fuse_packet,
    packetize_gaussians,
)
from ember.physical_team import PhysicalRobotTeam, select_source_packets
from ember.regions import RegionRobotTeam
from ember.robot_state import RobotState
from mcu import MCUOutOfMemory
from tests.test_robot_state import _graph


def packet(epoch=0, *, cell=0, sequence=0):
    return packetize_gaussians(
        0,
        sequence,
        np.array([cell]),
        np.array([2.0]),
        np.array([0.25]),
        map_epoch=epoch,
    )[0]


def test_epoch_zero_keeps_exact_legacy_wire_bytes():
    p = packet()
    expected = struct.pack("<BHBI", 0, 0, 4, 1) + struct.pack(
        "<Iii", 0, 8 * 65536, 4 * 65536
    )
    assert p.encode() == expected
    assert GaussianPacket.decode(expected).map_epoch == 0


@pytest.mark.parametrize("epoch", (1, 42, MAX_MAP_EPOCH))
def test_extended_epoch_roundtrip_forwarding_and_mtu(epoch):
    p = packetize_gaussians(
        0,
        2,
        np.arange(MAX_UPDATES),
        np.ones(MAX_UPDATES),
        np.ones(MAX_UPDATES),
        map_epoch=epoch,
    )[0]
    assert p.header_bytes == 16 and p.size_bytes == len(p.encode()) <= MAX_PAYLOAD_BYTES
    q = GaussianPacket.decode(p.encode())
    assert q.map_epoch == epoch
    np.testing.assert_array_equal(q.information_q16, p.information_q16)
    assert q.forwarded().map_epoch == epoch


@pytest.mark.parametrize("epoch", (-1, MAX_MAP_EPOCH + 1, 1.5, True))
def test_invalid_epoch_rejected(epoch):
    with pytest.raises(ValueError, match="epoch"):
        packet(epoch)
    with pytest.raises(ValueError, match="epoch"):
        RobotState(0, _graph(), np.zeros(3), map_epoch=epoch)


def test_extended_zero_epoch_is_rejected_as_noncanonical():
    p = packet(1)
    data = p.encode()[:12] + bytes(4) + p.encode()[16:]
    with pytest.raises(ValueError, match="nonzero"):
        GaussianPacket.decode(data)


def test_epoch_header_is_charged_when_selecting_partial_source_packet():
    p = packetize_gaussians(0, 0, np.arange(3), np.ones(3), np.ones(3), map_epoch=9)[0]
    args = ((p,), np.arange(3), np.array([1.0, 2.0, 3.0]))
    assert select_source_packets(*args, 27) == ()
    selected = select_source_packets(*args, 28)
    assert len(selected) == 1 and selected[0].size_bytes == 28
    assert selected[0].map_epoch == 9 and selected[0].cells.tolist() == [2]


@pytest.mark.parametrize("latest", (False, True))
def test_wrong_epoch_rejected_before_bounds_or_full_queue(latest):
    robot = RobotState(
        0, _graph(), np.zeros(3), map_epoch=2, queue_capacity=1, latest_evidence=latest
    )
    before = robot.engine.local_precision.copy()
    assert robot.enqueue(packet(2))
    assert not robot.enqueue(packet(1, cell=1000))
    assert not robot.enqueue(packet(3, cell=1000))
    assert robot.rejected_epoch_packets == robot.rejected_epoch_factors == 2
    assert len(robot.queue) == 1
    np.testing.assert_array_equal(before, robot.engine.local_precision)
    assert robot.consume() == 1


@pytest.mark.parametrize("latest", (False, True))
def test_queued_epoch_rechecked_before_fusion(latest):
    robot = RobotState(0, _graph(), np.zeros(3), latest_evidence=latest)
    robot.enqueue(packet())
    before = robot.engine.local_precision.copy()
    robot.map_epoch = 1
    assert robot.consume() == 0 and not robot.seen_factors
    assert robot.rejected_epoch_packets == 1 and not robot.queue
    np.testing.assert_array_equal(before, robot.engine.local_precision)


def test_stateless_fusion_requires_explicit_matching_map():
    belief = GaussianBelief.uniform(3, mean=0.0, variance=1.0)
    with pytest.raises(ValueError, match="epoch"):
        fuse_packet(belief, packet(1))
    assert fuse_packet(belief, packet(1), map_epoch=1).mean[0] == pytest.approx(1.6)


@pytest.mark.parametrize("regional", (False, True))
@pytest.mark.parametrize("latest", (False, True))
def test_rebuild_rejects_delayed_frames_and_current_epoch_keeps_learning(
    regional, latest
):
    g = _graph()
    cls = RegionRobotTeam if regional else PhysicalRobotTeam
    team = cls(
        g,
        g.world_positions()[[0, 2]],
        loss_seed=1,
        latest_evidence=latest,
        **({"region_cells": 1} if regional else {}),
    )
    old = team.robots[0]
    delayed = old.observe(np.array([0]), np.array([2.0]), np.array([0.25]))[0]
    # Remove the old first voxel: index zero now names the old second voxel.
    rebuilt = replace(
        g,
        cells=g.cells[1:],
        surface_z=g.surface_z[1:],
        edges=np.array([[0, 1]]),
        edge_cost=np.ones(1),
    )
    team.replace_graph(rebuilt, process_variance=0)
    current = team.robots[0]
    assert team.map_epoch == current.map_epoch == 1
    before = current.engine.local_precision.copy()
    assert not current.enqueue(delayed)
    assert current.consume() == 0
    np.testing.assert_array_equal(before, current.engine.local_precision)
    fresh = current.observe(np.array([0]), np.array([3.0]), np.array([0.25]))[0]
    assert fresh.map_epoch == 1 and fresh.sequence > delayed.sequence
    assert current.consume() == 1
    team.replace_graph(rebuilt, process_variance=0)
    newest = team.robots[0]
    assert newest.map_epoch == 2 and newest.rejected_epoch_packets == 1
    assert not newest.enqueue(fresh)


@pytest.mark.parametrize("regional", (False, True))
def test_failed_preparation_preserves_map_epoch_evidence_and_pending_queue(
    regional, monkeypatch
):
    g = _graph()
    cls = RegionRobotTeam if regional else PhysicalRobotTeam
    team = cls(
        g,
        g.world_positions()[[0, 2]],
        loss_seed=1,
        **({"region_cells": 1} if regional else {}),
    )
    robots = team.robots
    original_graph = team.graph
    original_regions = getattr(team, "regions", None)
    robots[0].observe(np.array([0]), np.array([2.0]), np.array([0.25]))
    before = robots[0].engine.local_precision.copy()
    changed = replace(g, surface_z=g.surface_z + 0.01)

    def fail(*args):
        raise MCUOutOfMemory("injected preparation failure")

    monkeypatch.setattr(team, "_new_robot", fail)
    with pytest.raises(MCUOutOfMemory):
        team.replace_graph(changed)
    assert team.map_epoch == team.geometry_revisions == 0
    assert team.graph is original_graph and team.robots is robots
    assert getattr(team, "regions", None) is original_regions
    assert len(robots[0].queue) == 1 and not robots[0].seen_factors
    np.testing.assert_array_equal(before, robots[0].engine.local_precision)


@pytest.mark.parametrize("regional", (False, True))
def test_exhausted_epoch_never_wraps_or_consumes_pending_factors(regional):
    g = _graph()
    cls = RegionRobotTeam if regional else PhysicalRobotTeam
    team = cls(
        g,
        g.world_positions()[[0, 2]],
        loss_seed=1,
        **({"region_cells": 1} if regional else {}),
    )
    robots = team.robots
    robots[0].observe(np.array([0]), np.array([2.0]), np.array([0.25]))
    team.map_epoch = MAX_MAP_EPOCH
    with pytest.raises(OverflowError, match="epoch"):
        team.replace_graph(g)
    assert team.map_epoch == MAX_MAP_EPOCH and team.robots is robots
    assert len(robots[0].queue) == 1 and team.geometry_revisions == 0


def test_radio_flood_preserves_epoch_and_charges_every_extended_header():
    from ember.radio import RadioConfig, RadioMedium

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    results = []
    for epoch in (0, 3):
        medium = RadioMedium(positions, config=RadioConfig(packet_error_rate=0))
        result = medium.flood_packet(0, packet(epoch), np.random.default_rng(7))
        assert all(received for received in result.received)
        assert all(
            frame.map_epoch == epoch for frames in result.received for frame in frames
        )
        results.append(result)
    a, b = results
    assert a.stats.broadcasts == b.stats.broadcasts
    assert b.stats.bytes_air - a.stats.bytes_air == 8 * b.stats.broadcasts


@pytest.mark.parametrize("latest", (False, True))
def test_queued_wrong_epoch_cannot_raise_regional_hotspot_floor(latest):
    g = _graph()
    team = RegionRobotTeam(
        g,
        g.world_positions()[[0, 2]],
        loss_seed=1,
        region_cells=1,
        latest_evidence=latest,
    )
    robot = team.robots[0]
    robot.enqueue(packet())
    robot.map_epoch = 1
    assert robot.consume() == 0
    np.testing.assert_array_equal(robot.hazard_floor, np.zeros(robot.graph.n_nodes))


@pytest.mark.parametrize("cut", (1, 4, 8, 13, 16))
def test_truncated_epoch_frame_never_aliases_legacy(cut):
    encoded = packet(1).encode()
    with pytest.raises(ValueError):
        GaussianPacket.decode(encoded[:-cut])


def test_zero_provenance_is_not_a_valid_legacy_factor():
    with pytest.raises(ValueError, match="provenance"):
        replace(packet(), provenance=0)
