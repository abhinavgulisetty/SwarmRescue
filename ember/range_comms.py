"""Source RSSI frames for one fixed localization pose epoch."""

from __future__ import annotations

import struct
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np

from ember.comms import MAX_PAYLOAD_BYTES

# Both zero provenance words are reserved and invalid in Gaussian factor frames.
# An explicit magic and count prevent truncation from changing frame meaning.
_HEADER = struct.Struct("<BHBIII4sB")
_UPDATE = struct.Struct("<Hf")
_MAGIC = b"RNG1"
RANGE_HEADER_BYTES = _HEADER.size
MAX_RANGE_UPDATES = (MAX_PAYLOAD_BYTES - RANGE_HEADER_BYTES) // _UPDATE.size


def uint(value, maximum, name):
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or not 0 <= value <= maximum
    ):
        raise ValueError(f"{name} must be an unsigned bounded integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class RangePacket:
    sender: int
    sequence: int
    ttl: int
    pose_epoch: int
    beacons: np.ndarray
    rssi_dbm: np.ndarray

    def __post_init__(self):
        uint(self.sender, 255, "sender")
        uint(self.sequence, 65535, "sequence")
        uint(self.ttl, 255, "TTL")
        uint(self.pose_epoch, 0xFFFFFFFF, "pose epoch")
        beacons = np.asarray(self.beacons)
        readings = np.asarray(self.rssi_dbm, dtype=float)
        if (
            beacons.ndim != 1
            or not np.issubdtype(beacons.dtype, np.integer)
            or not 1 <= beacons.size <= MAX_RANGE_UPDATES
            or np.any(beacons < 0)
            or np.any(beacons > 65535)
            or len(np.unique(beacons)) != len(beacons)
        ):
            raise ValueError("beacons must be unique bounded uint16 IDs")
        if (
            readings.shape != beacons.shape
            or not np.all(np.isfinite(readings))
            or np.any(np.abs(readings) > np.finfo(np.float32).max)
        ):
            raise ValueError("RSSI readings must be finite matching float32 values")
        beacons = np.array(beacons, dtype=np.uint16, copy=True)
        readings = np.array(readings, dtype=np.float32, copy=True)
        beacons.setflags(write=False)
        readings.setflags(write=False)
        object.__setattr__(self, "beacons", beacons)
        object.__setattr__(self, "rssi_dbm", readings)

    @property
    def map_epoch(self):
        """Flood identity compatibility; this is a pose epoch, not a voxel map."""
        return self.pose_epoch

    @property
    def size_bytes(self):
        return RANGE_HEADER_BYTES + _UPDATE.size * len(self.beacons)

    def encode(self):
        payload = bytearray(
            _HEADER.pack(
                self.sender,
                self.sequence,
                self.ttl,
                0,
                0,
                self.pose_epoch,
                _MAGIC,
                len(self.beacons),
            )
        )
        for beacon, reading in zip(self.beacons, self.rssi_dbm, strict=True):
            payload.extend(_UPDATE.pack(int(beacon), float(reading)))
        return bytes(payload)

    @classmethod
    def decode(cls, payload):
        if not RANGE_HEADER_BYTES + _UPDATE.size <= len(payload) <= MAX_PAYLOAD_BYTES:
            raise ValueError("invalid range packet size")
        sender, sequence, ttl, zero1, zero2, epoch, magic, count = _HEADER.unpack_from(
            payload
        )
        if (
            zero1
            or zero2
            or magic != _MAGIC
            or len(payload) != RANGE_HEADER_BYTES + count * _UPDATE.size
        ):
            raise ValueError("invalid range frame marker, count or truncation")
        beacons, readings = zip(
            *struct.iter_unpack(_UPDATE.format, payload[RANGE_HEADER_BYTES:]),
            strict=True,
        )
        return cls(sender, sequence, ttl, epoch, np.array(beacons), np.array(readings))

    def forwarded(self):
        return replace(self, ttl=self.ttl - 1) if self.ttl else None


def packetize_ranges(sender, sequence, beacons, rssi_dbm, *, pose_epoch=0, ttl=2):
    ids = np.asarray(beacons)
    readings = np.asarray(rssi_dbm)
    if ids.ndim != 1 or ids.shape != readings.shape or not len(ids):
        raise ValueError("range vectors must be nonempty and matching")
    count = (len(ids) + MAX_RANGE_UPDATES - 1) // MAX_RANGE_UPDATES
    uint(sequence, 0xFFFFFFFFFFFFFFFF, "sequence")
    if sequence + count - 1 > 65535:
        raise OverflowError("range sequence exhausted; start a new session")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("beacon IDs must be unique across the source batch")
    return tuple(
        RangePacket(
            sender,
            sequence + i,
            ttl,
            pose_epoch,
            ids[offset : offset + MAX_RANGE_UPDATES],
            readings[offset : offset + MAX_RANGE_UPDATES],
        )
        for i, offset in enumerate(range(0, len(ids), MAX_RANGE_UPDATES))
    )
