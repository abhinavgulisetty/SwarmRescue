"""A small ESP-NOW-style broadcast medium and bounded flooding protocol.

The model deliberately stays below the belief layer: it moves encoded bytes,
applies path loss, wall attenuation, contention collisions and independent
packet loss, then lets callers decode the messages they actually received.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ember.comms import MAX_PAYLOAD_BYTES, GaussianPacket
from ember.range_comms import RangePacket


@dataclass(frozen=True, slots=True)
class RadioConfig:
    reference_rssi_dbm: float = -40.0
    path_loss_exponent: float = 2.2
    wall_loss_db: float = 7.0
    receiver_sensitivity_dbm: float = -88.0
    packet_error_rate: float = 0.02

    def __post_init__(self) -> None:
        values = (
            self.reference_rssi_dbm,
            self.path_loss_exponent,
            self.wall_loss_db,
            self.receiver_sensitivity_dbm,
            self.packet_error_rate,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("radio parameters must be finite")
        if self.path_loss_exponent <= 0.0 or self.wall_loss_db < 0.0:
            raise ValueError("path loss must be positive and wall loss nonnegative")
        if not 0.0 <= self.packet_error_rate <= 1.0:
            raise ValueError("packet_error_rate must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class Transmission:
    sender: int
    slot: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class RadioDelivery:
    receiver: int
    transmitter: int
    slot: int
    payload: bytes
    rssi_dbm: float


@dataclass(frozen=True, slots=True)
class RadioStats:
    broadcasts: int = 0
    attempted: int = 0
    delivered: int = 0
    collided: int = 0
    below_sensitivity: int = 0
    randomly_dropped: int = 0
    duplicates: int = 0
    bytes_air: int = 0

    def __add__(self, other: RadioStats) -> RadioStats:
        return RadioStats(
            broadcasts=self.broadcasts + other.broadcasts,
            attempted=self.attempted + other.attempted,
            delivered=self.delivered + other.delivered,
            collided=self.collided + other.collided,
            below_sensitivity=self.below_sensitivity + other.below_sensitivity,
            randomly_dropped=self.randomly_dropped + other.randomly_dropped,
            duplicates=self.duplicates + other.duplicates,
            bytes_air=self.bytes_air + other.bytes_air,
        )


@dataclass(frozen=True, slots=True)
class FloodResult:
    received: tuple[tuple[GaussianPacket | RangePacket, ...], ...]
    stats: RadioStats
    rounds: int


class RadioMedium:
    """Broadcast bytes among fixed nodes using a slotted collision model."""

    def __init__(
        self,
        positions_m: np.ndarray,
        *,
        wall_counts: np.ndarray | None = None,
        config: RadioConfig | None = None,
        loss_seed: int | None = None,
    ) -> None:
        positions = np.asarray(positions_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] not in (2, 3):
            raise ValueError("positions_m must have shape (N, 2) or (N, 3)")
        if positions.shape[0] < 2 or not np.all(np.isfinite(positions)):
            raise ValueError("the medium needs at least two finite node positions")
        walls = (
            np.zeros((positions.shape[0], positions.shape[0]), dtype=np.int64)
            if wall_counts is None
            else np.asarray(wall_counts)
        )
        if walls.shape != (positions.shape[0], positions.shape[0]):
            raise ValueError("wall_counts must be an N by N matrix")
        if not np.issubdtype(walls.dtype, np.integer) or np.any(walls < 0):
            raise ValueError("wall_counts must contain nonnegative integers")
        self.positions_m = positions.copy()
        self.wall_counts = walls.astype(np.int64, copy=True)
        if loss_seed is not None and (
            isinstance(loss_seed, bool)
            or not isinstance(loss_seed, int)
            or not 0 <= loss_seed < 2**64
        ):
            raise ValueError("loss_seed must be an unsigned 64-bit integer")
        self.config = config or RadioConfig()
        self.loss_seed = loss_seed
        self._next_flood_slot = 0

    @property
    def n_nodes(self) -> int:
        return int(self.positions_m.shape[0])

    def rssi(self, sender: int, receiver: int) -> float:
        self._check_node(sender)
        self._check_node(receiver)
        delta = self.positions_m[sender] - self.positions_m[receiver]
        distance = max(
            float(np.linalg.norm(delta)),
            1.0,
        )
        return float(
            self.config.reference_rssi_dbm
            - 10.0 * self.config.path_loss_exponent * math.log10(distance)
            - self.config.wall_loss_db * self.wall_counts[sender, receiver]
        )

    def transmit(
        self,
        transmissions: tuple[Transmission, ...] | list[Transmission],
        rng: np.random.Generator,
    ) -> tuple[tuple[RadioDelivery, ...], RadioStats]:
        """Resolve all broadcasts; two audible frames in one slot collide."""

        frames = tuple(transmissions)
        for frame in frames:
            self._check_node(frame.sender)
            if frame.slot < 0:
                raise ValueError("radio slots must be nonnegative")
            if not 1 <= len(frame.payload) <= MAX_PAYLOAD_BYTES:
                raise ValueError(f"radio payload must fit in {MAX_PAYLOAD_BYTES} bytes")

        delivered: list[RadioDelivery] = []
        attempted = collided = below = dropped = 0
        for receiver in range(self.n_nodes):
            receiver_slots = {
                frame.slot for frame in frames if frame.sender != receiver
            }
            for slot in sorted(receiver_slots):
                candidates: list[tuple[Transmission, float]] = []
                for frame in frames:
                    if frame.sender == receiver or frame.slot != slot:
                        continue
                    attempted += 1
                    strength = self.rssi(frame.sender, receiver)
                    if strength < self.config.receiver_sensitivity_dbm:
                        below += 1
                    else:
                        candidates.append((frame, strength))
                if len(candidates) > 1:
                    collided += len(candidates)
                    continue
                if not candidates:
                    continue
                frame, strength = candidates[0]
                if self.loss_seed is None:
                    loss_draw = rng.random()
                else:
                    # Stateless common random numbers for physical radio events.
                    # Packet content and method identity cannot change link noise.
                    key = struct.pack(
                        "<QQQQ", self.loss_seed, slot, frame.sender, receiver
                    )
                    digest = hashlib.blake2b(key, digest_size=8).digest()
                    loss_draw = (int.from_bytes(digest, "little") >> 11) / 2**53
                if loss_draw < self.config.packet_error_rate:
                    dropped += 1
                    continue
                delivered.append(
                    RadioDelivery(receiver, frame.sender, slot, frame.payload, strength)
                )
        stats = RadioStats(
            broadcasts=len(frames),
            attempted=attempted,
            delivered=len(delivered),
            collided=collided,
            below_sensitivity=below,
            randomly_dropped=dropped,
            bytes_air=sum(len(frame.payload) for frame in frames),
        )
        return tuple(delivered), stats

    def flood_packet(
        self,
        origin: int,
        packet: GaussianPacket | RangePacket,
        rng: np.random.Generator,
        *,
        decoder: Callable[
            [bytes], GaussianPacket | RangePacket
        ] = GaussianPacket.decode,
    ) -> FloodResult:
        """Flood one packet with per-node deduplication and TTL-limited relays."""

        self._check_node(origin)
        # Validate the selected wire format before reserving slots/transmitting.
        decoder(packet.encode())
        received: list[list[GaussianPacket | RangePacket]] = [
            [] for _ in range(self.n_nodes)
        ]
        received[origin].append(packet)
        seen: list[set[tuple[int, int, int]]] = [set() for _ in range(self.n_nodes)]
        seen[origin].add((packet.sender, packet.sequence, packet.map_epoch))
        frontier: list[tuple[int, GaussianPacket | RangePacket]] = [(origin, packet)]
        total = RadioStats()
        rounds = 0
        while frontier:
            transmissions = tuple(
                Transmission(
                    relay,
                    self._next_flood_slot
                    + rounds * self.n_nodes
                    + (relay + message.sequence) % self.n_nodes,
                    message.encode(),
                )
                for relay, message in frontier
            )
            deliveries, stats = self.transmit(transmissions, rng)
            total += stats
            next_frontier: list[tuple[int, GaussianPacket | RangePacket]] = []
            duplicates = 0
            for delivery in deliveries:
                message = decoder(delivery.payload)
                key = (message.sender, message.sequence, message.map_epoch)
                if key in seen[delivery.receiver]:
                    duplicates += 1
                    continue
                seen[delivery.receiver].add(key)
                received[delivery.receiver].append(message)
                forwarded = message.forwarded()
                if forwarded is not None:
                    next_frontier.append((delivery.receiver, forwarded))
            total += RadioStats(duplicates=duplicates)
            frontier = next_frontier
            rounds += 1
        self._next_flood_slot += rounds * self.n_nodes
        return FloodResult(tuple(tuple(items) for items in received), total, rounds)

    def _check_node(self, node: int) -> None:
        if not 0 <= node < self.n_nodes:
            raise ValueError("radio node is outside the medium")


__all__ = [
    "FloodResult",
    "RadioConfig",
    "RadioDelivery",
    "RadioMedium",
    "RadioStats",
    "Transmission",
]
