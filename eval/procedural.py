"""Procedural multi-world trajectory evaluation.

This module deliberately leaves the fixed tactical smoke test in
:mod:`eval.harness` unchanged.  Its experimental unit is a paired
``(world_seed, noise_seed)`` scenario.  Cascade owns world generation while
stable evaluation substreams own sensor, radio, localization, and
method-internal randomness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise, permutations
from pathlib import Path
from typing import Any, Literal

import numpy as np

from cascade.extract import TraversabilityGraph, VoxelGrid
from cascade.floorplan import FloorPlan, build_floorplan
from cascade.params import BuildingParams
from cascade.rng import stage_rng
from ember.baselines import last_received_mean_map
from ember.belief import GaussianBelief, observe_state_cost, sample_hazard
from ember.comms import GaussianPacket, fuse_packet, packetize_gaussians
from ember.consensus import gaussian_average_consensus
from ember.gbp import GBPConfig
from ember.localization import (
    LocalizationEstimate,
    RSSIConfig,
    localize_rssi,
    sample_rssi,
)
from ember.manifest import capture_manifest, write_runtime_source_bundle
from ember.radio import RadioConfig, RadioMedium, RadioStats
from ember.robot_state import RobotState
from ember.routing import (
    LMDPConfig,
    build_lmdp,
    oracle_state_cost,
)
from ember.routing.deadline import deadline_route, safeguard_route_deadline
from ember.voi import (
    TokenBucket,
    schedule_coverage_updates,
    schedule_gaussian_updates,
    solve_adjoint,
    solve_policy_occupancy,
)
from eval.harness import parse_seed_spec
from hazard import HazardConfig, HazardDomain, HazardSolver, HazardState
from mcu import Q16_SCALE
from mcu.log_lmdp import solve_log_q16

DEVELOPMENT_WORLD_SEEDS = frozenset(range(200))
HELD_OUT_WORLD_SEEDS = frozenset(range(10_000, 10_050))
PROCEDURAL_METHOD_ORDER = (
    "b1_shortest",
    "b2_flooding_dijkstra",
    "b3_distributed_consensus",
    "ember",
)
WORLD_STREAMS = (
    "floorplan",
    "collapse",
    "hazard",
    "survivors",
    "search",
)
NOISE_STREAMS = (
    "sensor_noise",
    "radio_loss",
    "ble_noise",
    "b1_internal",
    "b2_internal",
    "b3_internal",
    "ember_internal",
)
_NOISE_STREAM_INDEX = {name: index for index, name in enumerate(NOISE_STREAMS)}
_METHOD_INTERNAL_STREAM = {
    "b1_shortest": "b1_internal",
    "b2_flooding_dijkstra": "b2_internal",
    "b3_distributed_consensus": "b3_internal",
    "ember": "ember_internal",
}
PROCEDURAL_UNAVAILABLE_METRICS = (
    "iso_13571_survival",
    "physics_collapse_geometry",
    "ram_high_water_bytes",
    "mcu_cycles",
    "deadline_misses",
    "nca_sweeps",
)


@dataclass(frozen=True, slots=True)
class ProceduralEvaluationConfig:
    world_seeds: tuple[int, ...] = tuple(range(40))
    noise_seeds: tuple[int, ...] = (0, 1, 2)
    bootstrap_samples: int = 2_000
    confidence_level: float = 0.95
    bootstrap_seed: int = 0xC45CADE
    robot_count: int = 8
    survivor_count: int = 3
    observation_time_s: float = 5.0
    trajectory_dt_s: float = 1.0
    hazard_horizon_s: float = 120.0
    robot_speed_m_s: float = 1.0
    walking_speed_m_s: float = 0.8
    allow_held_out: bool = False
    voi_packet_budget_bytes: int = 64
    lmdp_control_temperature: float = 0.25
    uncertainty_beta: float = 1.0
    communication_policy: str = "adjoint_kl"
    escort_policy: str = "deadline_lmdp"
    inference_policy: str = "gbp"

    def __post_init__(self) -> None:
        if self.communication_policy not in (
            "adjoint_kl", "kl_only", "occupancy_coverage",
        ):
            raise ValueError("unknown communication policy")
        if self.escort_policy not in (
            "lmdp", "risk_astar", "deadline_pareto", "deadline_lmdp"
        ):
            raise ValueError("unknown escort policy")
        if self.inference_policy not in ("gbp", "canonical"):
            raise ValueError("unknown inference policy")
        for name, seeds in (
            ("world_seeds", self.world_seeds),
            ("noise_seeds", self.noise_seeds),
        ):
            if not seeds:
                raise ValueError(f"{name} cannot be empty")
            if any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in seeds
            ):
                raise ValueError(f"{name} must contain nonnegative integers")
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"{name} must be unique")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.robot_count < 4:
            raise ValueError("robot_count must be at least four")
        if self.survivor_count < 3:
            raise ValueError("survivor_count must be at least three")
        if self.voi_packet_budget_bytes < 20:
            raise ValueError("VoI budget must fit one Gaussian update")
        for name, value in (
            ("observation_time_s", self.observation_time_s),
            ("trajectory_dt_s", self.trajectory_dt_s),
            ("hazard_horizon_s", self.hazard_horizon_s),
            ("robot_speed_m_s", self.robot_speed_m_s),
            ("walking_speed_m_s", self.walking_speed_m_s),
            ("lmdp_control_temperature", self.lmdp_control_temperature),
            ("uncertainty_beta", self.uncertainty_beta),
        ):
            if value <= 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and positive")
        if self.hazard_horizon_s < self.trajectory_dt_s:
            raise ValueError("hazard_horizon_s must cover at least one trajectory step")
        if self.observation_time_s >= self.hazard_horizon_s:
            raise ValueError("observation_time_s must precede the hazard horizon")
        sealed = HELD_OUT_WORLD_SEEDS.intersection(self.world_seeds)
        if sealed and not self.allow_held_out:
            raise ValueError(
                "held-out world seeds 10000..10049 are sealed; pass "
                "allow_held_out=True only after implementation and thresholds "
                "are frozen"
            )

    @property
    def split(self) -> str:
        selected = set(self.world_seeds)
        if selected <= DEVELOPMENT_WORLD_SEEDS:
            return "development"
        if selected <= HELD_OUT_WORLD_SEEDS:
            return "held_out"
        return "custom"


@dataclass(frozen=True, slots=True)
class ProceduralWorld:
    world_seed: int
    plan: FloorPlan
    graph: TraversabilityGraph
    domain: HazardDomain
    rescue_start_state: HazardState
    dynamic_cost: np.ndarray
    rescue_start_step: int
    exit_nodes: tuple[int, ...]
    survivor_nodes: tuple[int, ...]
    robot_nodes: tuple[int, ...]
    ignition_node: int
    blocked_edges: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class ExogenousStreams:
    world_seed: int
    noise_seed: int
    received_packets: tuple[GaussianPacket, ...]
    received_by_robot: tuple[tuple[GaussianPacket, ...], ...]
    radio_stats: RadioStats
    voi_received_packets: tuple[GaussianPacket, ...]
    voi_received_by_robot: tuple[tuple[GaussianPacket, ...], ...]
    voi_radio_stats: RadioStats
    localizations: tuple[LocalizationEstimate, ...]
    method_internal_seeds: dict[str, int]


@dataclass(frozen=True, slots=True)
class TrajectoryResult:
    survivor_node: int
    route: tuple[int, ...]
    rescued: bool
    contacted: bool
    path_length_m: float
    contact_time_s: float
    evacuation_time_s: float
    rescue_time_s: float
    dynamic_exposure_proxy: float
    unpenalized_exposure_proxy: float
    objective_cost: float


@dataclass(frozen=True, slots=True)
class PolicyResult:
    method: str
    trajectories: tuple[TrajectoryResult, ...]
    mean_dynamic_exposure_proxy: float
    mean_unpenalized_exposure_proxy: float
    mean_contact_time_s: float
    time_to_first_contact_s: float
    mean_evacuation_time_s: float
    mean_rescue_time_s: float
    rescue_completion_rate: float
    contact_completion_rate: float
    mean_objective_cost: float
    mean_clairvoyant_regret: float
    radio_bytes: int
    inference_sweeps: int
    routing_sweeps: int
    routing_labels: int = 0
    deadline_resource_fallbacks: int = 0
    deadline_infeasible_plans: int = 0
    deadline_interventions: int = 0


@dataclass(frozen=True, slots=True)
class SearchAssignment:
    """One localization-driven robot-to-survivor search assignment."""

    survivor_node: int
    estimated_node: int
    robot_node: int
    approach_route: tuple[int, ...]
    contact_time_s: float


@dataclass(frozen=True, slots=True)
class ProceduralScenarioRun:
    world_seed: int
    noise_seed: int
    world_fingerprint: str
    blocked_edges: int
    methods: dict[str, PolicyResult]
    localization_coverage_95: float
    localization_mean_error_m: float
    radio_deliveries: int
    wall_time_s: float


@dataclass(frozen=True, slots=True)
class ProceduralEvaluationReport:
    schema_version: int
    protocol: dict[str, Any]
    provenance: dict[str, Any]
    unavailable_metrics: tuple[str, ...]
    runs: tuple[ProceduralScenarioRun, ...]
    summary: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


def noise_rng(world_seed: int, noise_seed: int, stream: str) -> np.random.Generator:
    """Return one stable noise substream for a paired procedural scenario."""

    if stream not in _NOISE_STREAM_INDEX:
        raise KeyError(f"unknown procedural noise stream: {stream}")
    sequence = np.random.SeedSequence(
        [world_seed, noise_seed, _NOISE_STREAM_INDEX[stream], 0xE8B3]
    )
    return np.random.default_rng(sequence)


def _room_graph(
    plan: FloorPlan,
) -> tuple[TraversabilityGraph, tuple[tuple[int, int], ...]]:
    node_keys: list[tuple[int, int]] = []
    positions: list[tuple[float, float, float]] = []
    node_at: dict[tuple[int, int], int] = {}
    for storey in plan.storeys:
        for room_index, room in enumerate(storey.rooms):
            node = len(node_keys)
            node_keys.append((storey.index, room_index))
            node_at[(storey.index, room_index)] = node
            positions.append(
                (
                    room.rect.center[0],
                    room.rect.center[1],
                    (storey.index + 0.5) * plan.params.storey_height_m,
                )
            )

    edge_set: set[tuple[int, int]] = set()

    def add_edge(left: int, right: int) -> None:
        if left != right:
            edge_set.add((min(left, right), max(left, right)))

    for storey in plan.storeys:
        for door in storey.doors:
            add_edge(
                node_at[(storey.index, door.room_a)],
                node_at[(storey.index, door.room_b)],
            )
        for corridor in storey.corridors:
            touched = [
                node_at[(storey.index, room_index)]
                for room_index, room in enumerate(storey.rooms)
                if room.rect.overlaps(corridor)
            ]
            for target in touched[1:]:
                add_edge(touched[0], target)

    for connector in plan.connectors:
        vertical_nodes = []
        for storey in plan.storeys:
            matches = [
                node_at[(storey.index, room_index)]
                for room_index, room in enumerate(storey.rooms)
                if room.kind is connector.kind and room.bay == connector.bay
            ]
            if len(matches) == 1:
                vertical_nodes.append(matches[0])
        for lower, upper in pairwise(vertical_nodes):
            add_edge(lower, upper)

    position_array = np.asarray(positions, dtype=np.float64)
    cell_size = 0.25
    cells = np.zeros((len(node_keys), 3), dtype=np.int32)
    cells[:, 0] = np.maximum(
        0, np.rint(position_array[:, 0] / cell_size - 0.5).astype(np.int32)
    )
    cells[:, 1] = np.maximum(
        0, np.rint(position_array[:, 1] / cell_size - 0.5).astype(np.int32)
    )
    cells[:, 2] = np.asarray([key[0] for key in node_keys], dtype=np.int32)
    shape = (
        int(np.max(cells[:, 0])) + 1,
        int(np.max(cells[:, 1])) + 1,
        plan.params.n_storeys,
    )
    occupied = np.zeros(shape, dtype=bool)
    edges = np.asarray(sorted(edge_set), dtype=np.int32)
    if edges.size == 0:
        edges = np.zeros((0, 2), dtype=np.int32)
        edge_cost = np.zeros(0, dtype=np.float64)
    else:
        edge_cost = np.linalg.norm(
            position_array[edges[:, 0]] - position_array[edges[:, 1]], axis=1
        )
        edge_cost = np.maximum(edge_cost, 0.25)
    graph = TraversabilityGraph(
        grid=VoxelGrid(
            (0.0, 0.0, 0.0),
            (cell_size, cell_size, plan.params.storey_height_m),
            occupied,
            occupied.copy(),
        ),
        cells=cells,
        surface_z=position_array[:, 2],
        edges=edges,
        edge_cost=edge_cost,
    )
    return graph, tuple(node_keys)


def _exit_nodes(
    plan: FloorPlan,
    graph: TraversabilityGraph,
    node_keys: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    positions = graph.world_positions()
    ground = np.asarray(
        [index for index, (storey, _) in enumerate(node_keys) if storey == 0],
        dtype=np.int64,
    )
    chosen: list[int] = []
    for exit_ in plan.exits:
        point = np.asarray(exit_.center)
        order = ground[
            np.argsort(np.linalg.norm(positions[ground, :2] - point, axis=1))
        ]
        node = next(
            (int(item) for item in order if int(item) not in chosen), int(order[0])
        )
        chosen.append(node)
    return tuple(chosen)


def _adjacency(graph: TraversabilityGraph) -> list[list[tuple[int, float]]]:
    return graph.adjacency()


def _spanning_tree_edges(
    graph: TraversabilityGraph, roots: tuple[int, ...]
) -> set[tuple[int, int]]:
    adjacency = _adjacency(graph)
    seen = set(roots)
    queue = list(roots)
    tree: set[tuple[int, int]] = set()
    while queue:
        source = queue.pop(0)
        for target, _ in sorted(adjacency[source]):
            if target in seen:
                continue
            seen.add(target)
            queue.append(target)
            tree.add((min(source, target), max(source, target)))
    if len(seen) != graph.n_nodes:
        raise RuntimeError("procedural room graph is disconnected before blockage")
    return tree


def _with_seeded_blockage(
    graph: TraversabilityGraph,
    exits: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[TraversabilityGraph, int]:
    tree = _spanning_tree_edges(graph, exits)
    candidates = [
        index
        for index, edge in enumerate(graph.edges)
        if (min(int(edge[0]), int(edge[1])), max(int(edge[0]), int(edge[1])))
        not in tree
    ]
    removed = {index for index in candidates if rng.random() < 0.18}
    if candidates and not removed:
        removed.add(candidates[int(rng.integers(len(candidates)))])
    keep = np.asarray(
        [index not in removed for index in range(graph.n_edges)], dtype=bool
    )
    return (
        TraversabilityGraph(
            graph.grid,
            graph.cells,
            graph.surface_z,
            graph.edges[keep],
            graph.edge_cost[keep],
        ),
        len(removed),
    )


def _multi_source_distance(
    graph: TraversabilityGraph, sources: tuple[int, ...]
) -> np.ndarray:
    distance = np.full(graph.n_nodes, np.inf)
    queue: list[tuple[float, int]] = []
    for source in sources:
        distance[source] = 0.0
        heapq.heappush(queue, (0.0, source))
    for_distance = _adjacency(graph)
    while queue:
        current, node = heapq.heappop(queue)
        if current != distance[node]:
            continue
        for neighbour, edge_cost in for_distance[node]:
            candidate = current + edge_cost
            if candidate < distance[neighbour]:
                distance[neighbour] = candidate
                heapq.heappush(queue, (candidate, neighbour))
    return distance


def _farthest_nodes(
    graph: TraversabilityGraph,
    count: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    positions = graph.world_positions()
    first = int(rng.integers(graph.n_nodes))
    chosen = [first]
    while len(chosen) < min(count, graph.n_nodes):
        separation = np.min(
            np.linalg.norm(
                positions[:, None, :2] - positions[np.asarray(chosen)][None, :, :2],
                axis=2,
            ),
            axis=1,
        )
        separation[np.asarray(chosen)] = -np.inf
        chosen.append(int(np.argmax(separation)))
    return tuple(chosen)


def _world_fingerprint(
    graph: TraversabilityGraph,
    exits: tuple[int, ...],
    survivors: tuple[int, ...],
    ignition: int,
) -> str:
    digest = hashlib.sha256()
    for array in (graph.cells, graph.edges, graph.edge_cost, graph.surface_z):
        digest.update(np.asarray(array).tobytes())
    digest.update(np.asarray(exits, dtype=np.int64).tobytes())
    digest.update(np.asarray(survivors, dtype=np.int64).tobytes())
    digest.update(np.asarray((ignition,), dtype=np.int64).tobytes())
    return digest.hexdigest()


def build_procedural_world(
    world_seed: int,
    config: ProceduralEvaluationConfig,
) -> ProceduralWorld:
    """Build one immutable Cascade world and its shared hazard evolution."""

    plan = build_floorplan(world_seed, BuildingParams())
    base_graph, node_keys = _room_graph(plan)
    exits = _exit_nodes(plan, base_graph, node_keys)
    graph, blocked_edges = _with_seeded_blockage(
        base_graph, exits, stage_rng(world_seed, "collapse")
    )
    distance = _multi_source_distance(graph, exits)
    positions = graph.world_positions()

    non_exit = np.asarray(
        [node for node in range(graph.n_nodes) if node not in exits], dtype=np.int64
    )
    survivor_rng = stage_rng(world_seed, "survivors")
    survivor_candidates = non_exit[
        distance[non_exit] >= np.quantile(distance[non_exit], 0.55)
    ]
    jitter = survivor_rng.random(survivor_candidates.size)
    survivor_order = np.lexsort((jitter, -distance[survivor_candidates]))
    survivors = tuple(
        int(node)
        for node in survivor_candidates[survivor_order[: config.survivor_count]]
    )
    if len(survivors) != config.survivor_count:
        raise RuntimeError("procedural world cannot place the requested survivors")

    hazard_rng = stage_rng(world_seed, "hazard")
    hazard_target = survivors[int(hazard_rng.integers(len(survivors)))]
    shortest = _route_to_any(
        graph,
        hazard_target,
        exits,
        np.zeros(graph.n_nodes),
        astar=True,
    )
    ignition_candidates = shortest[1:-1]
    if not ignition_candidates:
        ignition_candidates = tuple(int(node) for node in non_exit)
    ignition = int(hazard_rng.choice(ignition_candidates))

    robots = _farthest_nodes(graph, config.robot_count, stage_rng(world_seed, "search"))
    hazard_config = HazardConfig(cell_size_m=1.0, tick_seconds=0.2)
    domain = HazardDomain.from_floorplan(plan, hazard_config)
    initial_state = domain.initial_state().ignite(
        domain,
        tuple(positions[ignition]),
        radius_m=2.4,
        temperature_k=950.0,
        smoke_kg_m3=0.25,
    )
    solver = HazardSolver(domain)
    step_ticks_float = config.trajectory_dt_s / hazard_config.tick_seconds
    if not math.isclose(step_ticks_float, round(step_ticks_float), abs_tol=1.0e-9):
        raise ValueError("trajectory_dt_s must be divisible by the hazard tick")
    step_ticks = round(step_ticks_float)
    observation_steps_float = config.observation_time_s / config.trajectory_dt_s
    if not math.isclose(
        observation_steps_float, round(observation_steps_float), abs_tol=1.0e-9
    ):
        raise ValueError("observation_time_s must be divisible by trajectory_dt_s")
    observation_step = round(observation_steps_float)
    horizon_steps = math.ceil(config.hazard_horizon_s / config.trajectory_dt_s)
    dynamic_cost = np.empty((horizon_steps + 1, graph.n_nodes), dtype=np.float64)
    dynamic_cost[0] = oracle_state_cost(graph, domain, initial_state)
    evolving = initial_state
    rescue_start_state = initial_state
    for step in range(1, horizon_steps + 1):
        evolving = solver.step(evolving, ticks=step_ticks)
        dynamic_cost[step] = oracle_state_cost(graph, domain, evolving)
        if step == observation_step:
            rescue_start_state = evolving
    dynamic_cost.setflags(write=False)

    return ProceduralWorld(
        world_seed=world_seed,
        plan=plan,
        graph=graph,
        domain=domain,
        rescue_start_state=rescue_start_state,
        dynamic_cost=dynamic_cost,
        rescue_start_step=observation_step,
        exit_nodes=exits,
        survivor_nodes=survivors,
        robot_nodes=robots,
        ignition_node=ignition,
        blocked_edges=blocked_edges,
        fingerprint=_world_fingerprint(graph, exits, survivors, ignition),
    )


def _observe_packets(
    world: ProceduralWorld,
    sensor_rng: np.random.Generator,
    radio_rng: np.random.Generator,
) -> tuple[
    tuple[GaussianPacket, ...],
    tuple[tuple[GaussianPacket, ...], ...],
    tuple[tuple[int, GaussianPacket], ...],
    RadioStats,
]:
    positions = world.graph.world_positions()
    robot_positions = positions[np.asarray(world.robot_nodes)]
    distance = np.linalg.norm(
        positions[:, None, :] - robot_positions[None, :, :], axis=2
    )
    observed = np.flatnonzero(np.min(distance, axis=1) <= 6.0)
    owner = np.argmin(distance[observed], axis=1)
    reading = sample_hazard(
        world.rescue_start_state,
        world.domain,
        positions[observed],
        sensor_rng,
    )
    mean, variance = observe_state_cost(reading)
    packets_by_sender: list[tuple[int, GaussianPacket]] = []
    for sender in range(len(world.robot_nodes)):
        selected = np.flatnonzero(owner == sender)
        if selected.size == 0:
            continue
        packets = packetize_gaussians(
            sender,
            0,
            observed[selected],
            mean[selected],
            variance[selected],
        )
        packets_by_sender.extend((sender, packet) for packet in packets)

    medium = RadioMedium(
        robot_positions,
        config=RadioConfig(packet_error_rate=0.05),
        loss_seed=int(radio_rng.integers(0, 2**63)),
    )
    received_by_robot: list[list[GaussianPacket]] = [
        [] for _ in world.robot_nodes
    ]
    stats = RadioStats()
    for sender, packet in packets_by_sender:
        result = medium.flood_packet(sender, packet, radio_rng)
        for receiver, packets in enumerate(result.received):
            received_by_robot[receiver].extend(packets)
        stats += result.stats
    central = tuple(received_by_robot[0])
    return (
        central,
        tuple(tuple(items) for items in received_by_robot),
        tuple(packets_by_sender),
        stats,
    )


def _schedule_voi_packets(
    world: ProceduralWorld,
    source_packets: tuple[tuple[int, GaussianPacket], ...],
    packet_budget_bytes: int,
    control_temperature: float,
    radio_rng: np.random.Generator,
    localizations: tuple[LocalizationEstimate, ...],
    communication_policy: str = "adjoint_kl",
) -> tuple[
    tuple[GaussianPacket, ...], RadioStats,
    tuple[tuple[GaussianPacket, ...], ...],
]:
    """Schedule source evidence by adjoint-weighted KL and flood selected packets."""

    model = build_lmdp(
        world.graph,
        np.full(world.graph.n_nodes, 0.30, dtype=np.float64),
        world.exit_nodes,
        config=LMDPConfig(
            control_temperature=control_temperature,
            self_loop_weight=0.0,
            minimum_state_cost=0.02,
            tolerance=1.0e-8,
            max_sweeps=4_000,
        ),
    )
    source_weight = np.zeros(model.n_states, dtype=np.float64)
    # Communication must use the same noisy estimates available to routing.
    # Multiple estimated survivors can occupy one graph node; preserve their mass.
    graph_positions = world.graph.world_positions()
    for estimate in localizations:
        estimated_node = int(
            np.argmin(np.linalg.norm(graph_positions - estimate.position_m, axis=1))
        )
        source_weight[estimated_node] += 1.0
    adjoint = solve_adjoint(model, source_weight)
    if not adjoint.converged:
        raise RuntimeError("VoI adjoint did not converge")

    positions = world.graph.world_positions()[
        np.asarray(world.robot_nodes, dtype=np.int64)
    ]
    medium = RadioMedium(
        positions,
        config=RadioConfig(packet_error_rate=0.05),
        loss_seed=int(radio_rng.integers(0, 2**63)),
    )
    received: list[list[GaussianPacket]] = [[] for _ in world.robot_nodes]
    # Local sensing is available even when the token bucket suppresses its broadcast.
    for origin, packet in source_packets:
        received[origin].append(packet)
    stats = RadioStats()
    for sender in range(len(world.robot_nodes)):
        own = tuple(
            packet for origin, packet in source_packets if origin == sender
        )
        if not own:
            continue
        cells = np.concatenate([packet.cells for packet in own]).astype(np.int64)
        information = np.concatenate(
            [packet.information_q16 for packet in own]
        ).astype(np.float64) / Q16_SCALE
        precision = np.concatenate(
            [packet.precision_q16 for packet in own]
        ).astype(np.float64) / Q16_SCALE
        posterior_mean = information / precision
        posterior_variance = 1.0 / precision
        relevance = adjoint.field
        if communication_policy == "kl_only":
            relevance = np.ones(model.n_states)
        elif communication_policy == "occupancy_coverage":
            local_belief = GaussianBelief.uniform(
                model.n_states, mean=0.30, variance=0.04
            )
            for packet in own:
                local_belief = fuse_packet(local_belief, packet)
            local_model = build_lmdp(
                world.graph, local_belief.routing_cost(1.0), world.exit_nodes,
                config=model.config,
            )
            local_solution = solve_log_q16(local_model)
            if not local_solution.fixed_point:
                raise RuntimeError("communication local routing did not converge")
            relevance = solve_policy_occupancy(
                local_model, local_solution.negative_log_desirability, source_weight
            )
        scheduler = (
            schedule_coverage_updates if communication_policy == "occupancy_coverage"
            else schedule_gaussian_updates
        )
        extra = (
            {"relevance": relevance, "positions_m": graph_positions}
            if communication_policy == "occupancy_coverage" else {"adjoint": relevance}
        )
        schedule = scheduler(
            sender=sender,
            sequence=1_000,
            cells=cells,
            prior_mean=np.full(cells.size, 0.30),
            prior_variance=np.full(cells.size, 0.04),
            posterior_mean=posterior_mean,
            posterior_variance=posterior_variance,
            **extra,
            bucket=TokenBucket(
                capacity_bytes=packet_budget_bytes,
                refill_bytes=0,
            ),
        )
        for packet in schedule.packets:
            result = medium.flood_packet(sender, packet, radio_rng)
            for receiver, packets in enumerate(result.received):
                if receiver != sender:
                    received[receiver].extend(packets)
            stats += result.stats
    by_robot = tuple(tuple(packets) for packets in received)
    return by_robot[0], stats, by_robot


def sample_exogenous_streams(
    world: ProceduralWorld,
    noise_seed: int,
    *,
    voi_packet_budget_bytes: int = 64,
    lmdp_control_temperature: float = 0.25,
    communication_policy: str = "adjoint_kl",
) -> ExogenousStreams:
    """Sample paired sensor/radio/BLE outcomes without mutating the world."""

    packets, packets_by_robot, source_packets, stats = _observe_packets(
        world,
        noise_rng(world.world_seed, noise_seed, "sensor_noise"),
        noise_rng(world.world_seed, noise_seed, "radio_loss"),
    )
    positions = world.graph.world_positions()
    robot_positions = positions[np.asarray(world.robot_nodes)]
    ble_rng = noise_rng(world.world_seed, noise_seed, "ble_noise")
    rssi_config = RSSIConfig(noise_std_db=0.45)
    localizations = []
    for survivor in world.survivor_nodes:
        truth = positions[survivor]
        rssi = sample_rssi(robot_positions, truth, ble_rng, config=rssi_config)
        localizations.append(localize_rssi(robot_positions, rssi, config=rssi_config))
    voi_packets, voi_stats, voi_by_robot = _schedule_voi_packets(
        world,
        source_packets,
        voi_packet_budget_bytes,
        lmdp_control_temperature,
        noise_rng(world.world_seed, noise_seed, "radio_loss"),
        tuple(localizations),
        communication_policy,
    )
    method_seeds = {
        method: int(
            noise_rng(
                world.world_seed,
                noise_seed,
                _METHOD_INTERNAL_STREAM[method],
            ).integers(0, np.iinfo(np.int64).max)
        )
        for method in PROCEDURAL_METHOD_ORDER
    }
    return ExogenousStreams(
        world_seed=world.world_seed,
        noise_seed=noise_seed,
        received_packets=packets,
        received_by_robot=packets_by_robot,
        radio_stats=stats,
        voi_received_packets=voi_packets,
        voi_received_by_robot=voi_by_robot,
        voi_radio_stats=voi_stats,
        localizations=tuple(localizations),
        method_internal_seeds=method_seeds,
    )


def _route_to_any(
    graph: TraversabilityGraph,
    start: int,
    goals: tuple[int, ...],
    state_cost: np.ndarray,
    *,
    astar: bool,
) -> tuple[int, ...]:
    costs = np.asarray(state_cost, dtype=np.float64)
    if costs.shape != (graph.n_nodes,) or np.any(costs < 0.0):
        raise ValueError("state cost must be nonnegative with one value per node")
    goal_set = set(goals)
    if start in goal_set:
        return (start,)
    positions = graph.world_positions()
    goal_positions = positions[np.asarray(goals)]

    def heuristic(node: int) -> float:
        if not astar:
            return 0.0
        return float(np.min(np.linalg.norm(goal_positions - positions[node], axis=1)))

    distance = np.full(graph.n_nodes, np.inf)
    previous = np.full(graph.n_nodes, -1, dtype=np.int64)
    distance[start] = 0.0
    queue: list[tuple[float, float, int]] = [(heuristic(start), 0.0, start)]
    reached = -1
    adjacency = _adjacency(graph)
    while queue:
        _, current_distance, node = heapq.heappop(queue)
        if current_distance != distance[node]:
            continue
        if node in goal_set:
            reached = node
            break
        for neighbour, edge_cost in adjacency[node]:
            candidate = current_distance + edge_cost + float(costs[neighbour])
            if candidate < distance[neighbour]:
                distance[neighbour] = candidate
                previous[neighbour] = node
                heapq.heappush(
                    queue,
                    (candidate + heuristic(neighbour), candidate, neighbour),
                )
    if reached < 0:
        return ()
    route = [reached]
    while route[-1] != start:
        route.append(int(previous[route[-1]]))
    route.reverse()
    return tuple(route)


def _edge_cost_lookup(graph: TraversabilityGraph) -> dict[tuple[int, int], float]:
    return {
        (min(int(left), int(right)), max(int(left), int(right))): float(cost)
        for (left, right), cost in zip(graph.edges, graph.edge_cost, strict=True)
    }


def _route_length(graph: TraversabilityGraph, route: tuple[int, ...]) -> float:
    lookup = _edge_cost_lookup(graph)
    return float(
        sum(
            lookup[(min(source, target), max(source, target))]
            for source, target in pairwise(route)
        )
    )


def _blind_depth_first_search_route(
    graph: TraversabilityGraph,
    start: int,
    target: int,
) -> tuple[int, ...]:
    """Walk a deterministic DFS until contact, without target-informed ordering."""

    if not 0 <= start < graph.n_nodes or not 0 <= target < graph.n_nodes:
        raise ValueError("search endpoints must be graph nodes")
    if start == target:
        return (start,)
    neighbours = tuple(
        tuple(sorted(neighbour for neighbour, _ in adjacent))
        for adjacent in _adjacency(graph)
    )
    visited = {start}
    walk = [start]
    stack: list[tuple[int, int]] = [(start, 0)]
    while stack:
        node, offset = stack[-1]
        if offset >= len(neighbours[node]):
            stack.pop()
            if stack:
                walk.append(stack[-1][0])
            continue
        neighbour = neighbours[node][offset]
        stack[-1] = (node, offset + 1)
        if neighbour in visited:
            continue
        visited.add(neighbour)
        walk.append(neighbour)
        if neighbour == target:
            return tuple(walk)
        stack.append((neighbour, 0))
    raise RuntimeError("survivor is unreachable from the localization estimate")


def _search_assignments(
    world: ProceduralWorld,
    streams: ExogenousStreams,
    config: ProceduralEvaluationConfig,
) -> tuple[SearchAssignment, ...]:
    """Assign robots using BLE estimates, then charge localization misses."""

    positions = world.graph.world_positions()
    estimated_nodes = tuple(
        int(
            np.argmin(
                np.linalg.norm(
                    positions[:, : estimate.dimension] - estimate.position_m[None, :],
                    axis=1,
                )
            )
        )
        for estimate in streams.localizations
    )
    zero = np.zeros(world.graph.n_nodes)
    assignment_cost = np.full(
        (len(world.robot_nodes), len(world.survivor_nodes)),
        np.inf,
    )
    routes_to_estimate: dict[tuple[int, int], tuple[int, ...]] = {}
    for robot_index, robot_node in enumerate(world.robot_nodes):
        for survivor_index, estimated_node in enumerate(estimated_nodes):
            route = _route_to_any(
                world.graph,
                robot_node,
                (estimated_node,),
                zero,
                astar=True,
            )
            if route:
                routes_to_estimate[(robot_index, survivor_index)] = route
                assignment_cost[robot_index, survivor_index] = _route_length(
                    world.graph, route
                )

    assignment_options = (
        (
            float(
                sum(
                    assignment_cost[robot_index, survivor_index]
                    for survivor_index, robot_index in enumerate(robot_indices)
                )
            ),
            robot_indices,
        )
        for robot_indices in permutations(
            range(len(world.robot_nodes)), len(world.survivor_nodes)
        )
    )
    best_cost, robot_for_survivor = min(assignment_options)
    if not math.isfinite(best_cost):
        raise RuntimeError("not enough reachable robots for survivor search")

    assignments = []
    for survivor_index, survivor_node in enumerate(world.survivor_nodes):
        robot_index = int(robot_for_survivor[survivor_index])
        estimated_node = estimated_nodes[survivor_index]
        approach = routes_to_estimate[(robot_index, survivor_index)]
        local_search = _blind_depth_first_search_route(
            world.graph, estimated_node, survivor_node
        )
        combined = approach + local_search[1:]
        approach_steps = math.ceil(
            _route_length(world.graph, combined)
            / config.robot_speed_m_s
            / config.trajectory_dt_s
        )
        contact_time = (
            world.rescue_start_step + approach_steps
        ) * config.trajectory_dt_s
        assignments.append(
            SearchAssignment(
                survivor_node=survivor_node,
                estimated_node=estimated_node,
                robot_node=world.robot_nodes[robot_index],
                approach_route=combined,
                contact_time_s=contact_time,
            )
        )
    return tuple(assignments)


def _evaluate_trajectory(
    world: ProceduralWorld,
    survivor: int,
    route: tuple[int, ...],
    config: ProceduralEvaluationConfig,
    *,
    contact_time_s: float,
) -> TrajectoryResult:
    contact_step = math.ceil(contact_time_s / config.trajectory_dt_s)
    horizon = world.dynamic_cost.shape[0] - 1
    contacted = contact_step <= horizon
    contact_time = min(contact_step, horizon) * config.trajectory_dt_s
    waiting_stop = min(contact_step, horizon)
    waiting_dose = float(
        np.sum(world.dynamic_cost[1 : waiting_stop + 1, survivor])
        * config.trajectory_dt_s
    )

    def failed(path_length: float, exposure: float) -> TrajectoryResult:
        penalty = config.hazard_horizon_s + 1_000.0
        return TrajectoryResult(
            survivor_node=survivor,
            route=route,
            rescued=False,
            path_length_m=path_length,
            contact_time_s=contact_time,
            evacuation_time_s=max(0.0, config.hazard_horizon_s - contact_time),
            contacted=contacted,
            rescue_time_s=config.hazard_horizon_s,
            dynamic_exposure_proxy=exposure + penalty,
            unpenalized_exposure_proxy=exposure,
            objective_cost=2.0 * penalty,
        )

    if not route or route[0] != survivor or route[-1] not in world.exit_nodes:
        return failed(0.0, waiting_dose)
    if contact_step >= horizon:
        return failed(0.0, waiting_dose)
    lookup = _edge_cost_lookup(world.graph)
    elapsed_steps = contact_step
    path_length = 0.0
    dose = waiting_dose
    for source, target in pairwise(route):
        length = lookup[(min(source, target), max(source, target))]
        duration = length / config.walking_speed_m_s
        steps = max(1, math.ceil(duration / config.trajectory_dt_s))
        elapsed_steps += steps
        if elapsed_steps > horizon:
            return failed(path_length, dose)
        discrete_duration = steps * config.trajectory_dt_s
        dose += float(world.dynamic_cost[elapsed_steps, target]) * discrete_duration
        path_length += length
    rescue_time = elapsed_steps * config.trajectory_dt_s
    return TrajectoryResult(
        survivor_node=survivor,
        route=route,
        rescued=True,
        path_length_m=path_length,
        contact_time_s=contact_time,
        evacuation_time_s=rescue_time - contact_time,
        rescue_time_s=rescue_time,
        contacted=True,
        dynamic_exposure_proxy=dose,
        unpenalized_exposure_proxy=dose,
        objective_cost=rescue_time + dose,
    )


def _time_expanded_oracle(
    world: ProceduralWorld,
    survivor: int,
    config: ProceduralEvaluationConfig,
    *,
    contact_time_s: float,
) -> TrajectoryResult:
    goals = set(world.exit_nodes)
    lookup = _edge_cost_lookup(world.graph)
    adjacency = _adjacency(world.graph)
    horizon = world.dynamic_cost.shape[0] - 1
    contact_step = math.ceil(contact_time_s / config.trajectory_dt_s)
    contact_time = contact_step * config.trajectory_dt_s
    waiting_dose = float(
        np.sum(world.dynamic_cost[1 : min(contact_step, horizon) + 1, survivor])
        * config.trajectory_dt_s
    )
    start = (contact_step, survivor)
    initial_objective = contact_time + waiting_dose
    distance: dict[tuple[int, int], float] = {start: initial_objective}
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    queue: list[tuple[float, int, int]] = [(initial_objective, contact_step, survivor)]
    reached: tuple[int, int] | None = None
    while queue:
        objective, step, node = heapq.heappop(queue)
        state = (step, node)
        if objective != distance[state]:
            continue
        if node in goals:
            reached = state
            break
        for neighbour, edge_length in adjacency[node]:
            duration = edge_length / config.walking_speed_m_s
            delta = max(1, math.ceil(duration / config.trajectory_dt_s))
            arrival = step + delta
            if arrival > horizon:
                continue
            discrete_duration = delta * config.trajectory_dt_s
            candidate = (
                objective
                + discrete_duration
                + float(world.dynamic_cost[arrival, neighbour]) * discrete_duration
            )
            next_state = (arrival, neighbour)
            if candidate < distance.get(next_state, math.inf):
                distance[next_state] = candidate
                previous[next_state] = state
                heapq.heappush(queue, (candidate, arrival, neighbour))
    if reached is None:
        return _evaluate_trajectory(
            world,
            survivor,
            (),
            config,
            contact_time_s=contact_time,
        )
    states = [reached]
    while states[-1] != start:
        states.append(previous[states[-1]])
    states.reverse()
    route = tuple(node for _, node in states)
    result = _evaluate_trajectory(
        world,
        survivor,
        route,
        config,
        contact_time_s=contact_time,
    )
    expected = distance[reached]
    if not math.isclose(
        result.objective_cost, expected, rel_tol=1.0e-10, abs_tol=1.0e-10
    ):
        raise AssertionError("time-expanded oracle objective mismatch")
    for left, right in pairwise(route):
        if (min(left, right), max(left, right)) not in lookup:
            raise AssertionError("oracle route contains a non-edge")
    return result


def _belief_from_packets(
    world: ProceduralWorld,
    packets: tuple[GaussianPacket, ...],
) -> GaussianBelief:
    belief = GaussianBelief.uniform(world.graph.n_nodes, mean=0.30, variance=0.04)
    for packet in packets:
        belief = fuse_packet(belief, packet)
    return belief


def run_policy(
    scenario: ProceduralWorld,
    method: Literal[
        "b1_shortest",
        "b2_flooding_dijkstra",
        "b3_distributed_consensus",
        "ember",
    ],
    exogenous_streams: ExogenousStreams,
    config: ProceduralEvaluationConfig,
) -> PolicyResult:
    """Run one policy from a shared immutable world and exogenous realization."""

    if scenario.world_seed != exogenous_streams.world_seed:
        raise ValueError("world and exogenous streams have different world seeds")
    inference_sweeps = 0
    routing_sweeps = 0
    routing_labels = deadline_resource_fallbacks = deadline_infeasible_plans = 0
    deadline_interventions = 0
    routes: list[tuple[int, ...]] = []
    zero = np.zeros(scenario.graph.n_nodes)
    assignments = _search_assignments(scenario, exogenous_streams, config)

    if method == "b1_shortest":
        routes = [
            _route_to_any(
                scenario.graph, survivor, scenario.exit_nodes, zero, astar=True
            )
            for survivor in scenario.survivor_nodes
        ]
    elif method == "b2_flooding_dijkstra":
        mean = last_received_mean_map(
            scenario.graph.n_nodes,
            exogenous_streams.received_packets,
            prior_mean=0.30,
        )
        routes = [
            _route_to_any(
                scenario.graph, survivor, scenario.exit_nodes, mean, astar=False
            )
            for survivor in scenario.survivor_nodes
        ]
    elif method == "b3_distributed_consensus":
        robot_count = len(scenario.robot_nodes)
        local_beliefs = []
        for robot, packets in enumerate(exogenous_streams.received_by_robot):
            belief = GaussianBelief.uniform(
                scenario.graph.n_nodes,
                mean=0.30,
                variance=0.04 * robot_count,
            )
            for packet in packets:
                if packet.sender == robot:
                    belief = fuse_packet(belief, packet)
            local_beliefs.append(belief)
        positions = scenario.graph.world_positions()[
            np.asarray(scenario.robot_nodes, dtype=np.int64)
        ]
        medium = RadioMedium(positions, config=RadioConfig(packet_error_rate=0.05))
        adjacency = np.zeros((robot_count, robot_count), dtype=bool)
        for left in range(robot_count):
            for right in range(left + 1, robot_count):
                linked = (
                    medium.rssi(left, right)
                    >= medium.config.receiver_sensitivity_dbm
                )
                adjacency[left, right] = adjacency[right, left] = linked
        consensus = gaussian_average_consensus(local_beliefs, adjacency)
        inference_sweeps = consensus.rounds
        robot_index = {
            node: index for index, node in enumerate(scenario.robot_nodes)
        }
        routes = []
        for survivor, assignment in zip(
            scenario.survivor_nodes, assignments, strict=True
        ):
            belief = consensus.beliefs[robot_index[assignment.robot_node]]
            risk = belief.mean + np.sqrt(belief.variance)
            routes.append(
                _route_to_any(
                    scenario.graph,
                    survivor,
                    scenario.exit_nodes,
                    risk,
                    astar=True,
                )
            )
    elif method == "ember":
        robots = tuple(
            RobotState(
                robot, scenario.graph,
                scenario.graph.world_positions()[node],
                gbp_config=GBPConfig(
                    edge_variance=0.8, damping=0.6,
                    tolerance=1.0e-6, max_sweeps=1_000,
                ),
            )
            for robot, node in enumerate(scenario.robot_nodes)
        )
        for robot, packets in zip(
            robots, exogenous_streams.voi_received_by_robot, strict=True
        ):
            for packet in packets:
                robot.enqueue(packet)
        # Interleave robot ticks. Each receiver owns its messages and evidence.
        if config.inference_policy == "canonical":
            for robot in robots:
                robot.consume()
        pending = (
            set(range(len(robots))) if config.inference_policy == "gbp" else set()
        )
        for _ in range(1_000):
            for index in sorted(pending):
                robot = robots[index]
                if robot.tick() and robot.last_residual <= robot.gbp_config.tolerance:
                    pending.remove(index)
            if not pending:
                break
        if pending:
            raise RuntimeError("distributed robot inference did not converge")
        inference_sweeps = max(robot.inference_sweeps for robot in robots)
        robot_index = {
            node: index for index, node in enumerate(scenario.robot_nodes)
        }
        routes = []
        for survivor, assignment in zip(
            scenario.survivor_nodes, assignments, strict=True
        ):
            robot = robots[robot_index[assignment.robot_node]]
            risk_cost = robot.engine.belief().routing_cost(config.uncertainty_beta)
            if config.escort_policy == "risk_astar":
                routes.append(_route_to_any(
                    scenario.graph, survivor, scenario.exit_nodes, risk_cost, astar=True
                ))
                continue
            if config.escort_policy == "deadline_pareto":
                remaining = max(0, math.ceil(
                    config.hazard_horizon_s / config.trajectory_dt_s
                ) - math.ceil(assignment.contact_time_s / config.trajectory_dt_s))
                constrained = deadline_route(
                    scenario.graph, survivor, scenario.exit_nodes, risk_cost,
                    available_steps=remaining,
                    walking_speed_m_s=config.walking_speed_m_s,
                    dt_s=config.trajectory_dt_s,
                )
                routing_labels += constrained.labels_used
                deadline_resource_fallbacks += int(constrained.resource_fallback)
                deadline_infeasible_plans += int(not constrained.feasible)
                routes.append(constrained.route)
                continue
            model = build_lmdp(
                scenario.graph,
                risk_cost,
                scenario.exit_nodes,
                config=LMDPConfig(
                    control_temperature=config.lmdp_control_temperature,
                    self_loop_weight=0.0, minimum_state_cost=0.02,
                    tolerance=1.0e-7, max_sweeps=4_000,
                ),
            )
            solution = robot.plan(model)
            if not solution.fixed_point and config.escort_policy != "deadline_lmdp":
                raise RuntimeError("robot integer routing did not converge")
            routing_sweeps = max(routing_sweeps, solution.sweeps)
            proposal = solution.route(survivor) if solution.fixed_point else ()
            if config.escort_policy == "deadline_lmdp":
                remaining = max(0, math.ceil(
                    config.hazard_horizon_s / config.trajectory_dt_s
                ) - math.ceil(assignment.contact_time_s / config.trajectory_dt_s))
                constrained = safeguard_route_deadline(
                    scenario.graph, survivor, scenario.exit_nodes, risk_cost, proposal,
                    available_steps=remaining,
                    walking_speed_m_s=config.walking_speed_m_s,
                    dt_s=config.trajectory_dt_s, arena=robot.arena,
                )
                routing_labels += constrained.labels_used
                deadline_interventions += int(constrained.proposal_replaced)
                deadline_resource_fallbacks += int(constrained.resource_fallback)
                deadline_infeasible_plans += int(not constrained.feasible)
                routes.append(constrained.route)
            else:
                routes.append(proposal)
    else:
        raise ValueError(f"unknown procedural policy: {method}")

    trajectories = tuple(
        _evaluate_trajectory(
            scenario,
            survivor,
            route,
            config,
            contact_time_s=assignment.contact_time_s,
        )
        for survivor, route, assignment in zip(
            scenario.survivor_nodes, routes, assignments, strict=True
        )
    )
    oracle = tuple(
        _time_expanded_oracle(
            scenario,
            survivor,
            config,
            contact_time_s=assignment.contact_time_s,
        )
        for survivor, assignment in zip(
            scenario.survivor_nodes, assignments, strict=True
        )
    )
    objective = np.asarray([item.objective_cost for item in trajectories])
    oracle_objective = np.asarray([item.objective_cost for item in oracle])
    regret = objective - oracle_objective
    if np.any(regret < -1.0e-8):
        raise AssertionError("time-expanded clairvoyant regret cannot be negative")
    return PolicyResult(
        method=method,
        trajectories=trajectories,
        mean_unpenalized_exposure_proxy=float(np.mean([
            item.unpenalized_exposure_proxy for item in trajectories
        ])),
        mean_dynamic_exposure_proxy=float(
            np.mean([item.dynamic_exposure_proxy for item in trajectories])
        ),
        mean_contact_time_s=float(
            np.mean([item.contact_time_s for item in trajectories])
        ),
        time_to_first_contact_s=float(
            np.min([item.contact_time_s for item in trajectories])
        ),
        mean_evacuation_time_s=float(
            np.mean([item.evacuation_time_s for item in trajectories])
        ),
        mean_rescue_time_s=float(
            np.mean([item.rescue_time_s for item in trajectories])
        ),
        rescue_completion_rate=float(np.mean([item.rescued for item in trajectories])),
        contact_completion_rate=float(
            np.mean([item.contacted for item in trajectories])
        ),
        mean_objective_cost=float(np.mean(objective)),
        mean_clairvoyant_regret=float(np.mean(np.maximum(regret, 0.0))),
        radio_bytes=(
            0
            if method == "b1_shortest"
            else exogenous_streams.voi_radio_stats.bytes_air
            if method == "ember"
            else exogenous_streams.radio_stats.bytes_air
        ),
        inference_sweeps=inference_sweeps,
        routing_sweeps=routing_sweeps,
        routing_labels=routing_labels,
        deadline_resource_fallbacks=deadline_resource_fallbacks,
        deadline_infeasible_plans=deadline_infeasible_plans,
        deadline_interventions=deadline_interventions,
    )


def evaluate_procedural_scenario(
    world: ProceduralWorld,
    noise_seed: int,
    config: ProceduralEvaluationConfig,
) -> ProceduralScenarioRun:
    started = time.perf_counter()
    streams = sample_exogenous_streams(
        world,
        noise_seed,
        voi_packet_budget_bytes=config.voi_packet_budget_bytes,
        lmdp_control_temperature=config.lmdp_control_temperature,
        communication_policy=config.communication_policy,
    )
    methods = {
        method: run_policy(world, method, streams, config)
        for method in PROCEDURAL_METHOD_ORDER
    }
    positions = world.graph.world_positions()
    errors = []
    covered = []
    for survivor, estimate in zip(
        world.survivor_nodes, streams.localizations, strict=True
    ):
        truth = positions[survivor, : estimate.dimension]
        errors.append(float(np.linalg.norm(estimate.position_m - truth)))
        covered.append(estimate.covers_95(truth))
    return ProceduralScenarioRun(
        world_seed=world.world_seed,
        noise_seed=noise_seed,
        world_fingerprint=world.fingerprint,
        blocked_edges=world.blocked_edges,
        methods=methods,
        localization_coverage_95=float(np.mean(covered)),
        localization_mean_error_m=float(np.mean(errors)),
        radio_deliveries=streams.radio_stats.delivered,
        wall_time_s=time.perf_counter() - started,
    )


def _world_values(
    runs: Sequence[ProceduralScenarioRun],
    value,
) -> np.ndarray:
    grouped: dict[int, list[float]] = defaultdict(list)
    for run in runs:
        grouped[run.world_seed].append(float(value(run)))
    return np.asarray(
        [np.mean(grouped[seed]) for seed in sorted(grouped)],
        dtype=np.float64,
    )


def _world_bootstrap(
    values: np.ndarray,
    config: ProceduralEvaluationConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size == 0 or not np.all(np.isfinite(sample)):
        raise ValueError("world bootstrap input must be a nonempty finite vector")
    indices = rng.integers(
        0,
        sample.size,
        size=(config.bootstrap_samples, sample.size),
    )
    means = np.mean(sample[indices], axis=1)
    tail = (1.0 - config.confidence_level) / 2.0
    low, high = np.quantile(means, (tail, 1.0 - tail))
    return {
        "mean": float(np.mean(sample)),
        "minimum": float(np.min(sample)),
        "maximum": float(np.max(sample)),
        "ci_low": float(low),
        "ci_high": float(high),
        "world_count": int(sample.size),
    }


def _procedural_summary(
    runs: tuple[ProceduralScenarioRun, ...],
    config: ProceduralEvaluationConfig,
) -> dict[str, Any]:
    rng = np.random.default_rng(config.bootstrap_seed)
    attributes = (
        "mean_dynamic_exposure_proxy",
        "mean_contact_time_s",
        "time_to_first_contact_s",
        "mean_evacuation_time_s",
        "mean_rescue_time_s",
        "rescue_completion_rate",
        "contact_completion_rate",
        "mean_objective_cost",
        "mean_clairvoyant_regret",
        "radio_bytes",
        "inference_sweeps",
        "routing_sweeps",
        "mean_unpenalized_exposure_proxy",
        "routing_labels",
        "deadline_resource_fallbacks",
        "deadline_infeasible_plans",
        "deadline_interventions",
    )
    methods = {}
    for method in PROCEDURAL_METHOD_ORDER:
        methods[method] = {
            attribute: _world_bootstrap(
                _world_values(
                    runs,
                    lambda run, m=method, a=attribute: getattr(run.methods[m], a),
                ),
                config,
                rng,
            )
            for attribute in attributes
        }

    paired = {}
    for right in ("b2_flooding_dijkstra", "b3_distributed_consensus"):
        values = _world_values(
            runs,
            lambda run, r=right: (
                run.methods["ember"].mean_dynamic_exposure_proxy
                - run.methods[r].mean_dynamic_exposure_proxy
            ),
        )
        comparison = _world_bootstrap(values, config, rng)
        comparison.update(
            {
                "metric": "mean_dynamic_exposure_proxy",
                "left": "ember",
                "right": right,
                "world_wins": int(np.count_nonzero(values < -1.0e-12)),
                "world_ties": int(np.count_nonzero(np.abs(values) <= 1.0e-12)),
                "world_losses": int(np.count_nonzero(values > 1.0e-12)),
            }
        )
        paired[f"ember_minus_{right}"] = comparison

    system = {
        "localization_coverage_95": _world_bootstrap(
            _world_values(runs, lambda run: run.localization_coverage_95),
            config,
            rng,
        ),
        "localization_mean_error_m": _world_bootstrap(
            _world_values(runs, lambda run: run.localization_mean_error_m),
            config,
            rng,
        ),
        "blocked_edges": _world_bootstrap(
            _world_values(runs, lambda run: run.blocked_edges),
            config,
            rng,
        ),
        "scenario_wall_time_s": _world_bootstrap(
            _world_values(runs, lambda run: run.wall_time_s),
            config,
            rng,
        ),
    }
    radio_savings = _world_values(
        runs,
        lambda run: (
            run.methods["b2_flooding_dijkstra"].radio_bytes
            - run.methods["ember"].radio_bytes
        ),
    )
    communication = _world_bootstrap(radio_savings, config, rng)
    b2_radio = _world_values(
        runs,
        lambda run: run.methods["b2_flooding_dijkstra"].radio_bytes,
    )
    ember_radio = _world_values(
        runs,
        lambda run: run.methods["ember"].radio_bytes,
    )
    communication.update(
        {
            "metric": "radio_bytes_saved",
            "left": "ember_voi",
            "right": "b2_flooding",
            "mean_reduction_ratio": float(
                1.0 - np.mean(ember_radio) / np.mean(b2_radio)
            ),
        }
    )
    return {
        "methods": methods,
        "paired": paired,
        "system": system,
        "communication": communication,
        "claim_checks": {
            "b3_reference_evaluated": True,
            "b3_distributed_consensus_evaluated": True,
            "c3_voi_radio_evaluated": True,
            "c3_mean_radio_bytes_lower": communication["mean"] > 0.0,
            "world_varies_by_seed": True,
            "paper_superiority_claim_ready": False,
            "reason": (
                "Procedural evidence, distributed B3, and method-specific VoI radio "
                "are available, but ISO 13571 exposure and physics-derived collapse "
                "geometry are not implemented."
            ),
        },
    }


def evaluate_procedural(
    config: ProceduralEvaluationConfig,
    *,
    root: Path | None = None,
) -> ProceduralEvaluationReport:
    """Evaluate the Cartesian product of world and noise seeds."""

    worlds = {seed: build_procedural_world(seed, config) for seed in config.world_seeds}
    runs = tuple(
        evaluate_procedural_scenario(worlds[world_seed], noise_seed, config)
        for world_seed in config.world_seeds
        for noise_seed in config.noise_seeds
    )
    protocol = {
        "split": config.split,
        "scenario_family": "cascade_floorplan_room_graph",
        "experimental_unit": ["world_seed", "noise_seed"],
        "world_varies_by_seed": True,
        "world_seed_count": len(config.world_seeds),
        "noise_replicates_per_world": len(config.noise_seeds),
        "world_seeds": list(config.world_seeds),
        "noise_seeds": list(config.noise_seeds),
        "world_rng_streams": list(WORLD_STREAMS),
        "noise_rng_streams": list(NOISE_STREAMS),
        "paired_common_random_numbers": True,
        "radio_loss_coupling": "counter_based_absolute_slot_sender_receiver",
        "radio_loss_eventwise_paired": True,
        "voi_source_weight": "ble_estimate_nearest_graph_node",
        "voi_uses_survivor_truth": False,
        "ember_belief_state": "independent_per_robot_" + config.inference_policy,
        "ember_route_map": "assigned_robot_delivered_evidence",
        "ember_routing_arithmetic": (
            "integer_log_q16" if config.escort_policy in ("lmdp", "deadline_lmdp")
            else "integer_pareto_q16" if config.escort_policy == "deadline_pareto"
            else "float64_astar"
        ),
        "robot_tick_cycle_costs": "analytical_not_measured",
        "robot_metadata_ram_measured": False,
        "bootstrap_unit": "world_seed",
        "bootstrap_method": "world_cluster_mean_resampling",
        "bootstrap_samples": config.bootstrap_samples,
        "confidence_level": config.confidence_level,
        "bootstrap_seed": config.bootstrap_seed,
        "robot_count": config.robot_count,
        "survivor_count": config.survivor_count,
        "observation_time_s": config.observation_time_s,
        "trajectory_dt_s": config.trajectory_dt_s,
        "hazard_horizon_s": config.hazard_horizon_s,
        "ble_localization_dimensions": 3,
        "robot_speed_m_s": config.robot_speed_m_s,
        "walking_speed_m_s": config.walking_speed_m_s,
        "search_assignment": "ble_estimate_exact_minimum_distance_matching",
        "local_search_policy": "deterministic_blind_depth_first_graph_walk",
        "failure_times_censored_at_horizon": True,
        "b3_reference_implementation": "distributed_gaussian_average_consensus",
        "local_search_after_estimate": True,
        "survivor_waiting_exposure_included": True,
        "rescue_time_origin": "hazard_ignition",
        "dynamic_exposure": True,
        "dose_is_heat_smoke_proxy_not_iso_13571": True,
        "unpenalized_exposure_metric": "mean_unpenalized_exposure_proxy",
        "collapse_model": "seeded_connectivity_preserving_topology_blockage",
        "physics_collapse_used": False,
        "communication_policy_varies_by_method": True,
        "communication_efficiency_comparison_available": True,
        "ember_communication_policy": config.communication_policy,
        "ember_escort_policy": config.escort_policy,
        "ember_inference_policy": config.inference_policy,
        "exposure_metric_includes_failure_penalty": True,
        "failure_exposure_penalty": config.hazard_horizon_s + 1000.0,
        "voi_packet_budget_bytes_per_robot": config.voi_packet_budget_bytes,
        "lmdp_control_temperature": config.lmdp_control_temperature,
        "lmdp_temperature_selected_on": "development_worlds_0_through_19",
        "uncertainty_beta": config.uncertainty_beta,
    }
    manifest = capture_manifest(
        config.world_seeds[0],
        protocol,
        {"scenario_count": len(runs), "world_count": len(worlds)},
        root=root,
    )
    provenance = {
        "created_at_utc": manifest.created_at_utc,
        "git_sha": manifest.git_sha,
        "git_dirty": manifest.git_dirty,
        "dirty_diff_hash": manifest.dirty_diff_hash,
        "lockfile_hash": manifest.lockfile_hash,
        "config_hash": manifest.config_hash,
        "versions": manifest.versions,
    }
    return ProceduralEvaluationReport(
        schema_version=10,
        protocol=protocol,
        provenance=provenance,
        unavailable_metrics=PROCEDURAL_UNAVAILABLE_METRICS,
        runs=runs,
        summary=_procedural_summary(runs, config),
    )


def write_procedural_report(
    report: ProceduralEvaluationReport,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report.to_json(), encoding="utf-8")
    temporary.replace(path)


def write_procedural_csv(
    report: ProceduralEvaluationReport,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = (
        "world_seed",
        "noise_seed",
        "world_fingerprint",
        "method",
        "mean_dynamic_exposure_proxy",
        "mean_contact_time_s",
        "time_to_first_contact_s",
        "mean_evacuation_time_s",
        "mean_rescue_time_s",
        "rescue_completion_rate",
        "contact_completion_rate",
        "mean_objective_cost",
        "mean_clairvoyant_regret",
        "radio_bytes",
        "inference_sweeps",
        "routing_sweeps",
        "routing_labels",
        "mean_unpenalized_exposure_proxy",
        "deadline_resource_fallbacks",
        "deadline_infeasible_plans",
        "deadline_interventions",
        "localization_coverage_95",
        "localization_mean_error_m",
        "blocked_edges",
    )
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in report.runs:
            for method in PROCEDURAL_METHOD_ORDER:
                result = run.methods[method]
                writer.writerow(
                    {
                        "world_seed": run.world_seed,
                        "noise_seed": run.noise_seed,
                        "world_fingerprint": run.world_fingerprint,
                        "method": method,
                        "mean_dynamic_exposure_proxy": (
                            result.mean_dynamic_exposure_proxy
                        ),
                        "mean_contact_time_s": result.mean_contact_time_s,
                        "time_to_first_contact_s": result.time_to_first_contact_s,
                        "mean_evacuation_time_s": result.mean_evacuation_time_s,
                        "mean_rescue_time_s": result.mean_rescue_time_s,
                        "rescue_completion_rate": result.rescue_completion_rate,
                        "contact_completion_rate": result.contact_completion_rate,
                        "mean_objective_cost": result.mean_objective_cost,
                        "mean_clairvoyant_regret": result.mean_clairvoyant_regret,
                        "radio_bytes": result.radio_bytes,
                        "inference_sweeps": result.inference_sweeps,
                        "routing_sweeps": result.routing_sweeps,
                        "routing_labels": result.routing_labels,
                        "mean_unpenalized_exposure_proxy": (
                            result.mean_unpenalized_exposure_proxy
                        ),
                        "deadline_resource_fallbacks": (
                            result.deadline_resource_fallbacks
                        ),
                        "deadline_infeasible_plans": result.deadline_infeasible_plans,
                        "deadline_interventions": result.deadline_interventions,
                        "localization_coverage_95": run.localization_coverage_95,
                        "localization_mean_error_m": run.localization_mean_error_m,
                        "blocked_edges": run.blocked_edges,
                    }
                )
    temporary.replace(path)


def format_procedural_summary(report: ProceduralEvaluationReport) -> str:
    b2 = report.summary["paired"]["ember_minus_b2_flooding_dijkstra"]
    return "\n".join(
        (
            f"procedural scenarios: {len(report.runs)} "
            f"({report.protocol['world_seed_count']} worlds)",
            "EMBER - B2 dynamic exposure: "
            f"{b2['mean']:.6f} [{b2['ci_low']:.6f}, {b2['ci_high']:.6f}]",
            "world wins-ties-losses: "
            f"{b2['world_wins']}-{b2['world_ties']}-{b2['world_losses']}",
            "bootstrap unit: world_seed",
            "paper superiority claim: NOT READY",
            report.summary["claim_checks"]["reason"],
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.procedural",
        description="Procedural paired world/noise trajectory evaluation.",
    )
    parser.add_argument("--world-seeds", default="0:39")
    parser.add_argument("--noise-seeds", default="0:2")
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=0xC45CADE)
    parser.add_argument("--robot-count", type=int, default=8)
    parser.add_argument("--survivor-count", type=int, default=3)
    parser.add_argument("--hazard-horizon", type=float, default=120.0)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/eval/procedural.json"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/eval/procedural.csv"),
    )
    parser.add_argument("--communication-policy", default="adjoint_kl",
                        choices=("adjoint_kl", "kl_only", "occupancy_coverage"))
    parser.add_argument(
        "--escort-policy", default="deadline_lmdp",
        choices=("lmdp", "risk_astar", "deadline_pareto", "deadline_lmdp"),
    )
    parser.add_argument("--inference-policy", default="gbp",
                        choices=("gbp", "canonical"))
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--allow-held-out", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = ProceduralEvaluationConfig(
            world_seeds=parse_seed_spec(args.world_seeds),
            noise_seeds=parse_seed_spec(args.noise_seeds),
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence,
            bootstrap_seed=args.bootstrap_seed,
            robot_count=args.robot_count,
            survivor_count=args.survivor_count,
            hazard_horizon_s=args.hazard_horizon,
            allow_held_out=args.allow_held_out,
            communication_policy=args.communication_policy,
            escort_policy=args.escort_policy,
            inference_policy=args.inference_policy,
        )
    except ValueError as error:
        parser.error(str(error))

    report = evaluate_procedural(config)
    bundle_hash = None
    if args.source_bundle is not None:
        bundle_hash = write_runtime_source_bundle(args.source_bundle)
    write_procedural_report(report, args.out)
    write_procedural_csv(report, args.csv)
    print(format_procedural_summary(report))
    print(f"report: {args.out}")
    print(f"runs:   {args.csv}")
    if args.source_bundle is not None:
        print(f"source: {args.source_bundle} (sha256 {bundle_hash})")
    return 0


__all__ = [
    "DEVELOPMENT_WORLD_SEEDS",
    "HELD_OUT_WORLD_SEEDS",
    "NOISE_STREAMS",
    "PROCEDURAL_METHOD_ORDER",
    "WORLD_STREAMS",
    "ExogenousStreams",
    "PolicyResult",
    "ProceduralEvaluationConfig",
    "ProceduralEvaluationReport",
    "ProceduralScenarioRun",
    "ProceduralWorld",
    "TrajectoryResult",
    "build_procedural_world",
    "evaluate_procedural",
    "evaluate_procedural_scenario",
    "format_procedural_summary",
    "noise_rng",
    "run_policy",
    "sample_exogenous_streams",
    "write_procedural_csv",
    "write_procedural_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
