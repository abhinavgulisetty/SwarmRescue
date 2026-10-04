"""Gaussian radio packet and flooding-baseline contracts."""

from __future__ import annotations

import numpy as np
import pytest

from ember.belief import GaussianBelief
from ember.comms import (
    MAX_PAYLOAD_BYTES,
    MAX_UPDATES,
    GaussianPacket,
    fuse_packet,
    packetize_gaussians,
)


def test_packet_round_trip_stays_inside_esp_now_mtu() -> None:
    cells = np.arange(MAX_UPDATES, dtype=np.int64)
    packets = packetize_gaussians(
        3,
        41,
        cells,
        np.linspace(0.1, 0.9, cells.size),
        np.full(cells.size, 0.25),
    )

    assert len(packets) == 1
    payload = packets[0].encode()
    decoded = GaussianPacket.decode(payload)
    assert len(payload) <= MAX_PAYLOAD_BYTES
    assert decoded.sender == 3
    assert decoded.sequence == 41
    assert decoded.provenance == 1 << 3
    np.testing.assert_array_equal(decoded.cells, cells)


def test_packetizer_splits_updates_and_increments_sequence() -> None:
    count = 2 * MAX_UPDATES + 1
    packets = packetize_gaussians(
        1,
        65535,
        np.arange(count),
        np.full(count, 0.2),
        np.full(count, 0.1),
    )

    assert [packet.cells.size for packet in packets] == [MAX_UPDATES, MAX_UPDATES, 1]
    assert [packet.sequence for packet in packets] == [65535, 0, 1]


def test_received_q16_packet_fuses_gaussian_information() -> None:
    belief = GaussianBelief.uniform(2, mean=0.0, variance=4.0)
    packet = packetize_gaussians(
        0,
        0,
        np.asarray((1,)),
        np.asarray((2.0,)),
        np.asarray((1.0,)),
    )[0]

    posterior = fuse_packet(belief, GaussianPacket.decode(packet.encode()))

    assert posterior.mean[0] == 0.0
    assert posterior.mean[1] == pytest.approx(1.6)
    assert posterior.variance[1] == pytest.approx(0.8)


def test_ttl_expires_without_underflow() -> None:
    packet = packetize_gaussians(
        0,
        0,
        np.asarray((0,)),
        np.asarray((1.0,)),
        np.asarray((1.0,)),
        ttl=1,
    )[0]

    forwarded = packet.forwarded()
    assert forwarded is not None and forwarded.ttl == 0
    assert forwarded.forwarded() is None


def test_decoder_rejects_truncated_update() -> None:
    with pytest.raises(ValueError, match="truncated"):
        GaussianPacket.decode(bytes(21))
