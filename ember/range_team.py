"""Receiver-local range inference on actual byte-level radio delivery."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ember.localization import RSSIConfig, rssi_to_range
from ember.radio import RadioMedium, RadioStats
from ember.range_comms import RangePacket, packetize_ranges, uint
from ember.range_gbp import RangeFactorGBP, RangeGBPResult


@dataclass(frozen=True, slots=True)
class RangeTeamResult:
    solutions: tuple[RangeGBPResult | None, ...]
    failures: tuple[str | None, ...]
    radio_stats: RadioStats
    rounds: int
    factor_counts: tuple[int, ...]
    receiver_numeric_bytes: int
    team_numeric_bytes: int

    @property
    def available(self):
        return all(
            solution is not None and solution.converged for solution in self.solutions
        )


class RangeLocalizationTeam:
    """Static known pose priors, separate bounded likelihoods per receiver.

    Range packets carry measured RSSI, not posteriors or beacon truth. The pose
    priors and RSSI calibration are shared input assumptions. Assignments and
    anchor-prior transport are outside this layer. A new pose epoch requires a
    new team; mismatched frames cannot populate the current likelihood table.
    """

    def __init__(
        self,
        positions_m,
        pose_covariance_m2,
        n_beacons,
        radio: RadioMedium,
        *,
        pose_epoch=0,
        sensor_config=None,
        gbp_config=None,
    ):
        positions = np.asarray(positions_m, float)
        if (
            positions.ndim != 2
            or positions.shape[1] not in (2, 3)
            or not 3 <= len(positions) <= 32
            or len(positions) != radio.n_nodes
            or radio.positions_m.shape[1] < positions.shape[1]
            or not np.allclose(radio.positions_m[:, : positions.shape[1]], positions)
        ):
            raise ValueError(
                "range team requires matching radio poses for 3..32 robots"
            )
        if (
            not isinstance(n_beacons, int)
            or isinstance(n_beacons, bool)
            or not 1 <= n_beacons <= 32
        ):
            raise ValueError("range team supports 1..32 beacons")
        self.pose_epoch = uint(pose_epoch, 0xFFFFFFFF, "pose epoch")
        self.radio = radio
        self.sensor_config = sensor_config or RSSIConfig()
        self.receivers = tuple(
            RangeFactorGBP(positions, pose_covariance_m2, n_beacons, config=gbp_config)
            for _ in range(len(positions))
        )
        self.next_sequence = np.zeros(len(positions), dtype=np.uint32)
        self.rejected_epoch_packets = 0
        self.radio_stats = RadioStats()

    def _validate_likelihoods(self, packet):
        try:
            with np.errstate(over="raise", invalid="raise"):
                ranges = rssi_to_range(packet.rssi_dbm, config=self.sensor_config)
                sigma = np.maximum(
                    0.05,
                    math.log(10)
                    * ranges
                    * self.sensor_config.noise_std_db
                    / (10 * self.sensor_config.path_loss_exponent),
                )
                if not np.all(np.isfinite(ranges)) or not np.all(np.isfinite(sigma**2)):
                    raise ValueError("range likelihood is not representable")
        except FloatingPointError as error:
            raise ValueError("range likelihood is not representable") from error

    def ingest(self, receiver, packet):
        receiver = uint(receiver, len(self.receivers) - 1, "receiver")
        if not isinstance(packet, RangePacket):
            raise ValueError("range receiver requires a range packet")
        if packet.pose_epoch != self.pose_epoch:
            self.rejected_epoch_packets += 1
            return False
        engine = self.receivers[receiver]
        if packet.sender >= len(self.receivers) or np.any(
            packet.beacons >= engine.ranges.shape[1]
        ):
            raise ValueError("range source or beacon is outside the receiver graph")
        self._validate_likelihoods(packet)
        changed = False
        for beacon, reading in zip(packet.beacons, packet.rssi_dbm, strict=True):
            changed |= engine.observe_rssi(
                packet.sender,
                int(beacon),
                packet.sequence,
                float(reading),
                config=self.sensor_config,
            )
        return changed

    def _coverage_ready(self):
        dimension = self.receivers[0].anchors.shape[1]
        for engine in self.receivers:
            for beacon in range(engine.ranges.shape[1]):
                sources = np.flatnonzero(engine.sequences[:, beacon] >= 0)
                if len(sources) < dimension + 1:
                    return False
                anchors = engine.anchors[sources]
                if np.linalg.matrix_rank(anchors[1:] - anchors[0]) < dimension:
                    return False
        return True

    def exchange(self, readings_dbm, rng, *, max_rounds=3):
        max_rounds = uint(max_rounds, 8, "retransmission rounds")
        if max_rounds < 1:
            raise ValueError("at least one transmission round is required")
        readings = np.asarray(readings_dbm, float)
        n, b = self.receivers[0].ranges.shape
        if readings.shape != (n, b) or not np.all(np.isfinite(readings)):
            raise ValueError("one finite RSSI value is required per source/beacon")
        # Validate every source frame before mutating any sequence/evidence.
        batches = tuple(
            packetize_ranges(
                source,
                int(self.next_sequence[source]),
                np.arange(b),
                readings[source],
                pose_epoch=self.pose_epoch,
            )
            for source in range(n)
        )
        for packets in batches:
            for packet in packets:
                self._validate_likelihoods(packet)
        for source, packets in enumerate(batches):
            self.next_sequence[source] += len(packets)
            for packet in packets:
                self.ingest(source, RangePacket.decode(packet.encode()))
        stats = RadioStats()
        for _round_index in range(1, max_rounds + 1):
            for source, packets in enumerate(batches):
                for packet in packets:
                    flood = self.radio.flood_packet(
                        source, packet, rng, decoder=RangePacket.decode
                    )
                    stats += flood.stats
                    for receiver, received in enumerate(flood.received):
                        if receiver == source:
                            continue
                        for delivered in received:
                            self.ingest(receiver, delivered)
            if self._coverage_ready():
                break
        self.radio_stats += stats
        solutions, failures = [], []
        for engine in self.receivers:
            try:
                solution = engine.solve()
                solutions.append(solution)
                failures.append(
                    None if solution.converged else "range GBP did not converge"
                )
            except ValueError as error:
                solutions.append(None)
                failures.append(str(error))
        numeric = tuple(engine.retained_numeric_bytes for engine in self.receivers)
        return RangeTeamResult(
            tuple(solutions),
            tuple(failures),
            stats,
            _round_index,
            tuple(
                int(np.count_nonzero(engine.sequences >= 0))
                for engine in self.receivers
            ),
            max(numeric),
            sum(numeric) + self.next_sequence.nbytes,
        )
