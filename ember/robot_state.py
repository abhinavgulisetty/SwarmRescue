"""Independent robot inference, packet queues and bounded factor provenance.

Arena usage covers numeric GBP buffers. Python packet queues and provenance sets
are host reference metadata, and are not claimed as measured embedded RAM.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from cascade.extract import TraversabilityGraph
from ember.comms import GaussianPacket, packetize_gaussians
from ember.gbp import GaussianBeliefPropagation, GBPConfig
from mcu import Q16_SCALE, CycleBudget, MCUArena, MCUOutOfMemory
from mcu.log_lmdp import LogQ16Solution, solve_log_q16


@dataclass(slots=True)
class RobotState:
    robot_id: int
    graph: TraversabilityGraph
    position_m: np.ndarray
    arena: MCUArena = field(default_factory=lambda: MCUArena(200 * 1024))
    gbp_config: GBPConfig = field(default_factory=GBPConfig)
    queue_capacity: int = 256
    factor_capacity: int = 8192
    cycles_per_tick: int = 2_400_000
    drop_on_factor_overflow: bool = False
    dropped_factors: int = field(default=0, init=False)
    engine: GaussianBeliefPropagation = field(init=False)
    queue: deque[GaussianPacket] = field(default_factory=deque, init=False)
    seen_factors: set[tuple[int, int, int]] = field(default_factory=set, init=False)
    routing: LogQ16Solution | None = field(default=None, init=False)
    sensor_state: dict[str, float] = field(default_factory=dict, init=False)
    deadline_misses: int = field(default=0, init=False)
    inference_sweeps: int = field(default=0, init=False)
    last_residual: float = field(default=float("inf"), init=False)
    last_cycles: int = field(default=0, init=False)
    next_sequence: int = field(default=0, init=False)
    _routing_offset: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.robot_id < 32:
            raise ValueError("robot_id must fit the source provenance bitmap")
        if min(self.queue_capacity, self.factor_capacity, self.cycles_per_tick) < 1:
            raise ValueError("queue, factor and cycle capacities must be positive")
        self.position_m = np.array(self.position_m, dtype=float, copy=True)
        if self.position_m.shape != (3,) or not np.all(np.isfinite(self.position_m)):
            raise ValueError("robot position must be a finite 3D point")
        self.engine = GaussianBeliefPropagation(
            self.graph, mean=0.30, variance=0.04, config=self.gbp_config
        )
        # Retain each robot's numeric state in its own arena.
        for name in (
            "local_precision",
            "local_information",
            "source",
            "target",
            "reverse",
            "edge_variance",
            "message_precision",
            "message_information",
        ):
            source = getattr(self.engine, name)
            buffer = np.frombuffer(
                self.arena.alloc(source.nbytes, alignment=8), dtype=source.dtype
            )
            buffer[:] = source
            setattr(self.engine, name, buffer)
        self._routing_offset = self.arena.used

    def enqueue(self, packet: GaussianPacket) -> None:
        if len(self.queue) >= self.queue_capacity:
            raise BufferError("robot packet queue is full")
        if packet.sender >= 32 or not packet.provenance & (1 << packet.sender):
            raise ValueError("packet does not identify its source factor owner")
        if np.any(packet.cells >= self.graph.n_nodes):
            raise ValueError("packet cell is outside the robot graph")
        self.queue.append(packet)

    def observe(self, cells, means, variances) -> tuple[GaussianPacket, ...]:
        """Queue a new local sensor epoch without reusing factor identities."""
        packets = packetize_gaussians(
            self.robot_id, self.next_sequence, cells, means, variances
        )
        if self.next_sequence + len(packets) > 65536:
            raise BufferError("sensor factor sequence exhausted; start a new session")
        if len(self.queue) + len(packets) > self.queue_capacity:
            raise BufferError("robot packet queue is full")
        for packet in packets:
            self.enqueue(packet)
        self.next_sequence += len(packets)
        return packets

    def consume(self) -> int:
        """Fuse source/sequence/cell factors once, including relayed duplicates."""
        accepted = 0
        while self.queue:
            packet = self.queue[0]
            indices = []
            keys = set()
            for index, cell in enumerate(packet.cells):
                key = (packet.sender, packet.sequence, int(cell))
                if key not in self.seen_factors and key not in keys:
                    indices.append(index)
                    keys.add(key)
            if len(self.seen_factors) + len(keys) > self.factor_capacity:
                if not self.drop_on_factor_overflow:
                    raise BufferError("robot factor provenance capacity exceeded")
                # Reject the entire new packet. Never forget accepted IDs: doing
                # so would allow delayed duplicates to be fused twice.
                self.queue.popleft()
                self.dropped_factors += len(keys)
                continue
            self.queue.popleft()
            if not indices:
                continue
            self.engine.observe_canonical(
                packet.cells[indices].astype(np.int64),
                packet.information_q16[indices].astype(float) / Q16_SCALE,
                packet.precision_q16[indices].astype(float) / Q16_SCALE,
            )
            self.seen_factors.update(keys)
            accepted += len(indices)
        return accepted

    def tick(self, active_messages: np.ndarray | None = None) -> bool:
        """One local sweep; cycle costs are analytical, not MCU calibration."""
        budget = CycleBudget(self.cycles_per_tick)
        budget.charge(sum(packet.cells.size for packet in self.queue), cycles_each=40)
        budget.charge(self.graph.n_nodes, cycles_each=12)
        active_count = (
            self.engine.source.size
            if active_messages is None
            else int(np.count_nonzero(active_messages))
        )
        # All messages contribute to cavity totals; only active ones are updated.
        budget.charge(self.engine.source.size, cycles_each=16)
        budget.charge(active_count, cycles_each=64)
        self.last_cycles = budget.used
        if budget.missed:
            self.deadline_misses += 1
            return False
        self.consume()
        self.last_residual = self.engine.sweep(active_messages)
        self.inference_sweeps += 1
        return True

    def plan(self, model, *, native: bool = False) -> LogQ16Solution:
        """Reuse one arena tail for routing; old routing buffers are replaced."""
        # Reject oversized requests before overwriting the previous policy.
        required = 18 * model.n_states + 8 * model.col_idx.size + 12
        if self._routing_offset + required > self.arena.capacity:
            raise MCUOutOfMemory("robot routing workspace exceeds its arena")
        previous_used = self.arena.used
        self.arena.used = self._routing_offset
        try:
            if native:
                from mcu.native_log_lmdp import solve_native_log_q16

                self.routing = solve_native_log_q16(model, arena=self.arena)
            else:
                self.routing = solve_log_q16(model, arena=self.arena)
        except Exception:
            self.arena.used = previous_used
            raise
        return self.routing
