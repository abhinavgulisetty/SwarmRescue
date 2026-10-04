"""Tests for the lossy slotted radio and bounded flooding protocol."""

from __future__ import annotations

import numpy as np
import pytest

from ember.comms import MAX_PAYLOAD_BYTES, packetize_gaussians
from ember.radio import RadioConfig, RadioMedium, Transmission


def _packet(ttl: int = 3):
    return packetize_gaussians(
        0,
        7,
        np.asarray((4,)),
        np.asarray((0.8,)),
        np.asarray((0.1,)),
        ttl=ttl,
    )[0]


def test_simultaneous_audible_broadcasts_collide_at_third_receiver() -> None:
    medium = RadioMedium(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.5, 0.0))),
        config=RadioConfig(packet_error_rate=0.0),
    )
    frames = (
        Transmission(0, 2, _packet().encode()),
        Transmission(1, 2, _packet().encode()),
    )

    deliveries, stats = medium.transmit(frames, np.random.default_rng(3))

    assert not any(delivery.receiver == 2 for delivery in deliveries)
    assert stats.collided == 2


def test_wall_attenuation_can_put_a_link_below_sensitivity() -> None:
    walls = np.asarray(((0, 5), (5, 0)))
    medium = RadioMedium(
        np.asarray(((0.0, 0.0), (2.0, 0.0))),
        wall_counts=walls,
        config=RadioConfig(receiver_sensitivity_dbm=-70.0, packet_error_rate=0.0),
    )

    deliveries, stats = medium.transmit(
        (Transmission(0, 0, _packet().encode()),), np.random.default_rng(4)
    )

    assert not deliveries
    assert stats.below_sensitivity == 1


def test_flood_reaches_two_hop_node_once_and_decrements_ttl() -> None:
    medium = RadioMedium(
        np.asarray(((0.0, 0.0), (5.0, 0.0), (10.0, 0.0))),
        config=RadioConfig(
            path_loss_exponent=2.0,
            receiver_sensitivity_dbm=-56.0,
            packet_error_rate=0.0,
        ),
    )

    result = medium.flood_packet(0, _packet(ttl=2), np.random.default_rng(5))

    assert len(result.received[2]) == 1
    assert result.received[2][0].ttl == 1
    assert result.stats.duplicates > 0
    assert result.rounds <= 3


def test_radio_rejects_payload_above_esp_now_limit() -> None:
    medium = RadioMedium(np.asarray(((0.0, 0.0), (1.0, 0.0))))
    with pytest.raises(ValueError):
        medium.transmit(
            (Transmission(0, 0, bytes(MAX_PAYLOAD_BYTES + 1)),),
            np.random.default_rng(6),
        )


def test_counter_based_loss_is_independent_of_packet_order_and_rng() -> None:
    medium = RadioMedium(
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        config=RadioConfig(packet_error_rate=0.5), loss_seed=99,
    )
    target = Transmission(0, 13, _packet().encode())
    unrelated = Transmission(1, 3, _packet().encode())
    first, _ = medium.transmit([target], np.random.default_rng(3))
    second, _ = medium.transmit([unrelated, target], np.random.default_rng(89))
    assert first == tuple(delivery for delivery in second if delivery.slot == 13)
    # Different payloads in the same physical event receive the same loss draw.
    changed = Transmission(0, 13, _packet(ttl=1).encode())
    third, _ = medium.transmit([changed], np.random.default_rng(0))
    assert [delivery.receiver for delivery in first] == [
        delivery.receiver for delivery in third
    ]


def test_seeded_floods_advance_absolute_radio_time(monkeypatch) -> None:
    medium = RadioMedium(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        config=RadioConfig(packet_error_rate=0.0), loss_seed=99,
    )
    observed_slots = []
    original_transmit = medium.transmit

    def record(transmissions, rng):
        observed_slots.append([frame.slot for frame in transmissions])
        return original_transmit(transmissions, rng)

    monkeypatch.setattr(medium, "transmit", record)
    medium.flood_packet(0, _packet(ttl=0), np.random.default_rng(0))
    medium.flood_packet(0, _packet(ttl=0), np.random.default_rng(0))
    assert min(observed_slots[1]) > max(observed_slots[0])
