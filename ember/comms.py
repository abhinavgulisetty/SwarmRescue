"""Compact Gaussian belief packets for the Phase-1 flooding baseline."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ember.belief import GaussianBelief
from mcu import INT32_MAX, INT32_MIN, Q16_SCALE

MAX_PAYLOAD_BYTES = 244
_HEADER = struct.Struct("<BHBI")
_EPOCH_HEADER = struct.Struct("<BHBIII")
MAX_MAP_EPOCH = 0xFFFFFFFF
_UPDATE = struct.Struct("<Iii")
MAX_UPDATES = (MAX_PAYLOAD_BYTES - _HEADER.size) // _UPDATE.size
PACKET_HEADER_BYTES = _HEADER.size
UPDATE_BYTES = _UPDATE.size


def _q16(values: np.ndarray) -> np.ndarray:
    scaled = np.rint(np.asarray(values, dtype=np.float64) * Q16_SCALE)
    return np.clip(scaled, INT32_MIN, INT32_MAX).astype(np.int32)


@dataclass(frozen=True, slots=True)
class GaussianPacket:
    """Hazard radio message: canonical Gaussian updates plus provenance."""

    sender: int
    sequence: int
    ttl: int
    provenance: int
    cells: np.ndarray
    information_q16: np.ndarray
    precision_q16: np.ndarray
    map_epoch: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.map_epoch, Integral)
            or isinstance(self.map_epoch, bool)
            or not 0 <= self.map_epoch <= MAX_MAP_EPOCH
        ):
            raise ValueError("map epoch must fit in u32")
        if not 0 <= self.sender <= 255:
            raise ValueError("sender must fit in u8")
        if not 0 <= self.sequence <= 65535:
            raise ValueError("sequence must fit in u16")
        if not 0 <= self.ttl <= 255:
            raise ValueError("ttl must fit in u8")
        if not 1 <= self.provenance <= 0xFFFFFFFF:
            raise ValueError("provenance must identify at least one source in u32")
        cells = np.array(self.cells, dtype=np.uint32, copy=True)
        information = np.array(self.information_q16, dtype=np.int32, copy=True)
        precision = np.array(self.precision_q16, dtype=np.int32, copy=True)
        if cells.ndim != 1 or not 1 <= cells.size <= MAX_UPDATES:
            raise ValueError(f"a packet must contain 1..{MAX_UPDATES} updates")
        if information.shape != cells.shape or precision.shape != cells.shape:
            raise ValueError("Gaussian packet arrays must have matching shapes")
        if np.any(precision <= 0):
            raise ValueError("Gaussian precision must be positive")
        for values in (cells, information, precision):
            values.setflags(write=False)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "information_q16", information)
        object.__setattr__(self, "precision_q16", precision)

    @property
    def header_bytes(self) -> int:
        return _EPOCH_HEADER.size if self.map_epoch else _HEADER.size

    @property
    def size_bytes(self) -> int:
        return self.header_bytes + self.cells.size * _UPDATE.size

    def encode(self) -> bytes:
        fields = (self.sender, self.sequence, self.ttl, self.provenance)
        payload = bytearray(
            _EPOCH_HEADER.pack(
                self.sender, self.sequence, self.ttl, 0, self.provenance, self.map_epoch
            )
            if self.map_epoch
            else _HEADER.pack(*fields)
        )
        for cell, information, precision in zip(
            self.cells,
            self.information_q16,
            self.precision_q16,
            strict=True,
        ):
            payload.extend(_UPDATE.pack(int(cell), int(information), int(precision)))
        return bytes(payload)

    @classmethod
    def decode(cls, payload: bytes) -> GaussianPacket:
        if not _HEADER.size + _UPDATE.size <= len(payload) <= MAX_PAYLOAD_BYTES:
            raise ValueError("invalid Gaussian packet size")
        sender, sequence, ttl, marker = _HEADER.unpack_from(payload)
        # Zero source provenance is invalid for a factor packet, so reserve it
        # as an explicit extension marker. Truncation can never remove this
        # discriminator while preserving a valid prefix frame.
        header = _EPOCH_HEADER if marker == 0 else _HEADER
        if (
            len(payload) < header.size + _UPDATE.size
            or (len(payload) - header.size) % _UPDATE.size
        ):
            raise ValueError("truncated Gaussian update")
        if marker == 0:
            sender, sequence, ttl, _, provenance, epoch = header.unpack_from(payload)
            if epoch == 0:
                raise ValueError("extended packet requires a nonzero map epoch")
        else:
            provenance, epoch = marker, 0
        updates = list(struct.iter_unpack(_UPDATE.format, payload[header.size :]))
        cells, information, precision = zip(*updates, strict=True)
        return cls(
            sender,
            sequence,
            ttl,
            provenance,
            np.asarray(cells),
            np.asarray(information),
            np.asarray(precision),
            map_epoch=epoch,
        )

    def forwarded(self) -> GaussianPacket | None:
        if self.ttl == 0:
            return None
        return GaussianPacket(
            self.sender,
            self.sequence,
            self.ttl - 1,
            self.provenance,
            self.cells,
            self.information_q16,
            self.precision_q16,
            map_epoch=self.map_epoch,
        )


def packetize_gaussians(
    sender: int,
    sequence: int,
    cells: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    *,
    ttl: int = 4,
    provenance: int | None = None,
    map_epoch: int = 0,
) -> tuple[GaussianPacket, ...]:
    indices = np.asarray(cells)
    means = np.asarray(mean, dtype=np.float64)
    variances = np.asarray(variance, dtype=np.float64)
    if (
        indices.ndim != 1
        or indices.shape != means.shape
        or means.shape != variances.shape
    ):
        raise ValueError("Gaussian update vectors must have matching shapes")
    if indices.size == 0:
        return ()
    if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0):
        raise ValueError("cell indices must be nonnegative integers")
    if not np.all(np.isfinite(means)):
        raise ValueError("Gaussian means must be finite")
    if not np.all(np.isfinite(variances)) or np.any(variances <= 0.0):
        raise ValueError("Gaussian variances must be finite and positive")

    precision = 1.0 / variances
    information = means * precision
    source_bits = (1 << sender) if provenance is None and sender < 32 else provenance
    if source_bits is None:
        raise ValueError("senders above 31 need explicit provenance")
    packets = []
    for offset in range(0, indices.size, MAX_UPDATES):
        part = slice(offset, offset + MAX_UPDATES)
        packets.append(
            GaussianPacket(
                sender,
                (sequence + len(packets)) & 0xFFFF,
                ttl,
                source_bits,
                indices[part],
                _q16(information[part]),
                _q16(precision[part]),
                map_epoch=map_epoch,
            )
        )
    return tuple(packets)


def fuse_packet(
    belief: GaussianBelief,
    packet: GaussianPacket,
    *,
    map_epoch: int = 0,
) -> GaussianBelief:
    if packet.map_epoch != map_epoch:
        raise ValueError("packet map epoch does not match the belief map")
    precision = packet.precision_q16.astype(np.float64) / Q16_SCALE
    information = packet.information_q16.astype(np.float64) / Q16_SCALE
    return belief.update(
        packet.cells.astype(np.int64), information / precision, 1.0 / precision
    )


__all__ = [
    "MAX_MAP_EPOCH",
    "MAX_PAYLOAD_BYTES",
    "MAX_UPDATES",
    "PACKET_HEADER_BYTES",
    "UPDATE_BYTES",
    "GaussianPacket",
    "fuse_packet",
    "packetize_gaussians",
]
