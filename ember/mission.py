"""Small, fully computed Phase-1 rescue mission used by the live demo.

Unlike the large MuJoCo traversal, this tactical scenario runs in under a
second.  Every displayed result comes from the real hazard, sensor, Gaussian
belief, BLE localization, and LMDP implementations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cascade.extract import TraversabilityGraph, VoxelGrid
from ember.baselines import (
    BaselineResult,
    centralized_shortest_baseline,
    clairvoyant_dijkstra_baseline,
    evaluate_route,
    flooding_dijkstra_baseline,
    last_received_mean_map,
)
from ember.belief import observe_state_cost, sample_hazard
from ember.comms import GaussianPacket, packetize_gaussians
from ember.gbp import GaussianBeliefPropagation, GBPConfig, GBPResult
from ember.localization import (
    LocalizationEstimate,
    RSSIConfig,
    localize_rssi,
    sample_rssi,
)
from ember.radio import RadioConfig, RadioMedium, RadioStats
from ember.routing import (
    LMDPConfig,
    LMDPSolution,
    build_lmdp,
    greedy_route,
    oracle_state_cost,
    solve_jacobi,
)
from hazard import HazardConfig, HazardDomain, HazardSolver, HazardState
from mcu import Q16_SCALE

WIDTH = 18
HEIGHT = 10
START_XY = (16, 2)
EXIT_XY = (1, 2)
FIRE_XY = (9.5, 2.5)


@dataclass(frozen=True, slots=True)
class MissionWorld:
    graph: TraversabilityGraph
    domain: HazardDomain
    clear: HazardState
    walkable: np.ndarray
    node_at: dict[tuple[int, int], int]


@dataclass(frozen=True, slots=True)
class MissionResult:
    world: MissionWorld
    evolved: HazardState
    search_before: tuple[int, ...]
    escort_before: tuple[int, ...]
    search_after: tuple[int, ...]
    escort_after: tuple[int, ...]
    localization: LocalizationEstimate
    survivor_position_m: np.ndarray
    robot_positions_m: np.ndarray
    rssi_dbm: np.ndarray
    sensed_smoke_kg_m3: float
    peak_temperature_k: float
    routing_solution: LMDPSolution
    gbp_result: GBPResult
    gbp_sweeps: int
    centralized_baseline: BaselineResult
    flooding_baseline: BaselineResult
    clairvoyant_baseline: BaselineResult
    ember_result: BaselineResult
    shortest_dose: float
    ember_dose: float
    receiver_information_gain_nats: float
    radio_packets: int
    radio_bytes: int
    radio_stats: RadioStats

    @property
    def localization_error_m(self) -> float:
        return float(
            np.linalg.norm(self.localization.position_m - self.survivor_position_m)
        )

    @property
    def succeeded(self) -> bool:
        direct = self.world.node_at[(9, 2)]
        return (
            direct in self.search_before
            and direct not in self.escort_before
            and direct not in self.search_after
            and direct not in self.escort_after
            and self.localization_error_m < 1.0
            and self.ember_dose < self.shortest_dose
            and self.flooding_baseline.hazard_dose < self.shortest_dose
            and self.gbp_result.converged
            and self.radio_stats.delivered > 0
            and self.routing_solution.converged
        )


def build_mission_world() -> MissionWorld:
    walkable = np.ones((HEIGHT, WIDTH), dtype=bool)
    for x in (6, 11):
        walkable[:, x] = False
        walkable[2, x] = True
        walkable[7, x] = True

    coordinates = np.argwhere(walkable)
    cells = np.column_stack(
        (
            coordinates[:, 1],
            coordinates[:, 0],
            np.zeros(len(coordinates), dtype=np.int32),
        )
    )
    node_at = {(int(x), int(y)): index for index, (x, y, _) in enumerate(cells)}
    edges: list[tuple[int, int]] = []
    for (x, y), node in node_at.items():
        for neighbour in ((x + 1, y), (x, y + 1)):
            if neighbour in node_at:
                edges.append((node, node_at[neighbour]))

    occupied = np.zeros((WIDTH, HEIGHT, 1), dtype=bool)
    graph = TraversabilityGraph(
        grid=VoxelGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), occupied, occupied),
        cells=np.asarray(cells, dtype=np.int32),
        surface_z=np.full(len(cells), 0.5),
        edges=np.asarray(edges, dtype=np.int32),
        edge_cost=np.ones(len(edges)),
    )
    active = walkable[None, :, :]
    config = HazardConfig(cell_size_m=1.0, tick_seconds=0.1)
    domain = HazardDomain(
        origin_m=(0.0, 0.0, 0.0),
        bounds_max_m=(float(WIDTH), float(HEIGHT), 1.0),
        cell_size_m=1.0,
        active=active,
        connector=np.zeros_like(active),
        vertical_open=np.zeros_like(active),
        initial_fuel_kg_m3=active.astype(np.float64) * 0.35,
        config=config,
    )
    return MissionWorld(graph, domain, domain.initial_state(), walkable, node_at)


def _route(
    world: MissionWorld, cost: np.ndarray
) -> tuple[tuple[int, ...], LMDPSolution]:
    model = build_lmdp(
        world.graph,
        cost,
        (world.node_at[EXIT_XY],),
        config=LMDPConfig(
            control_temperature=0.1,
            self_loop_weight=0.0,
            minimum_state_cost=0.02,
            tolerance=1.0e-7,
            max_sweeps=4_000,
        ),
    )
    solution = solve_jacobi(model)
    return tuple(greedy_route(model, solution, world.node_at[START_XY])), solution


def _safe_route_nodes(world: MissionWorld) -> np.ndarray:
    safe = [
        node
        for (x, y), node in world.node_at.items()
        if (y == 7 and 1 <= x <= 16) or (x in (1, 16) and 2 <= y <= 7)
    ]
    return np.asarray(safe, dtype=np.int64)


def _observe_packets(
    gbp: GaussianBeliefPropagation, packets: tuple[GaussianPacket, ...]
) -> None:
    for packet in packets:
        gbp.observe_canonical(
            packet.cells.astype(np.int64),
            packet.information_q16.astype(np.float64) / Q16_SCALE,
            packet.precision_q16.astype(np.float64) / Q16_SCALE,
        )


def _flood_to(
    medium: RadioMedium,
    origin: int,
    receiver: int,
    packets: tuple[GaussianPacket, ...],
    rng: np.random.Generator,
) -> tuple[tuple[GaussianPacket, ...], RadioStats]:
    received: list[GaussianPacket] = []
    stats = RadioStats()
    for packet in packets:
        result = medium.flood_packet(origin, packet, rng)
        received.extend(result.received[receiver])
        stats += result.stats
    return tuple(received), stats


def run_mission(seed: int = 7) -> MissionResult:
    world = build_mission_world()
    rng = np.random.default_rng(seed)
    positions = world.graph.world_positions()

    survivor = np.asarray((16.5, 2.5))
    robots = np.asarray(((12.5, 1.5), (16.5, 1.5), (12.5, 6.5), (17.5, 6.5)))
    radio = RadioMedium(
        robots,
        config=RadioConfig(packet_error_rate=0.01),
    )
    gbp = GaussianBeliefPropagation(
        world.graph,
        mean=0.30,
        variance=0.04,
        config=GBPConfig(
            edge_variance=0.8,
            damping=0.6,
            tolerance=1.0e-7,
            max_sweeps=2_000,
        ),
    )

    safe_nodes = _safe_route_nodes(world)
    safe_reading = sample_hazard(world.clear, world.domain, positions[safe_nodes], rng)
    safe_mean, safe_variance = observe_state_cost(safe_reading)
    safe_packets = packetize_gaussians(0, 0, safe_nodes, safe_mean, safe_variance)
    safe_received, safe_radio = _flood_to(radio, 0, 1, safe_packets, rng)
    _observe_packets(gbp, safe_received)
    before_result = gbp.run()
    before = before_result.belief
    search_before, _ = _route(world, before.routing_cost(-2.0))
    escort_before, _ = _route(world, before.routing_cost(1.0))

    ignited = world.clear.ignite(
        world.domain,
        (*FIRE_XY, 0.5),
        radius_m=1.45,
        temperature_k=950.0,
        smoke_kg_m3=0.32,
    )
    evolved = HazardSolver(world.domain).step(ignited, ticks=16)
    fire_nodes = np.flatnonzero(
        np.linalg.norm(positions[:, :2] - np.asarray(FIRE_XY), axis=1) <= 2.6
    )
    fire_reading = sample_hazard(evolved, world.domain, positions[fire_nodes], rng)
    fire_mean, fire_variance = observe_state_cost(fire_reading)
    fire_packets = packetize_gaussians(2, 0, fire_nodes, fire_mean, fire_variance)
    fire_received, fire_radio = _flood_to(radio, 2, 1, fire_packets, rng)
    _observe_packets(gbp, fire_received)
    after_result = gbp.run()
    after = after_result.belief
    search_after, _ = _route(world, after.routing_cost(-2.0))
    escort_after, solution = _route(world, after.routing_cost(1.0))

    rssi_config = RSSIConfig(noise_std_db=0.45)
    rssi = sample_rssi(robots, survivor, rng, config=rssi_config)
    localization = localize_rssi(robots, rssi, config=rssi_config)

    truth_cost = oracle_state_cost(world.graph, world.domain, evolved)
    total_radio = safe_radio + fire_radio
    centralized = centralized_shortest_baseline(
        world.graph,
        world.node_at[START_XY],
        world.node_at[EXIT_XY],
        truth_cost,
    )
    flooding_mean = last_received_mean_map(
        world.graph.n_nodes,
        safe_received + fire_received,
        prior_mean=0.30,
    )
    flooding = flooding_dijkstra_baseline(
        world.graph,
        world.node_at[START_XY],
        world.node_at[EXIT_XY],
        flooding_mean,
        truth_cost,
        radio_bytes=total_radio.bytes_air,
    )
    clairvoyant = clairvoyant_dijkstra_baseline(
        world.graph,
        world.node_at[START_XY],
        world.node_at[EXIT_XY],
        truth_cost,
    )
    ember_result = evaluate_route(
        "EMBER LMDP",
        world.graph,
        escort_after,
        truth_cost,
        radio_bytes=total_radio.bytes_air,
    )
    # Receiver-side information gain from the actually delivered fire update.
    # Marginal KL is honest about the current scalar (not coupled) GBP model.
    information_gain = float(
        np.sum(
            0.5
            * (
                np.log(before.variance / after.variance)
                + (after.variance + (after.mean - before.mean) ** 2) / before.variance
                - 1.0
            )
        )
    )
    return MissionResult(
        world=world,
        evolved=evolved,
        search_before=search_before,
        escort_before=escort_before,
        search_after=search_after,
        escort_after=escort_after,
        localization=localization,
        survivor_position_m=survivor,
        robot_positions_m=robots,
        rssi_dbm=rssi,
        sensed_smoke_kg_m3=float(np.max(fire_reading.smoke_kg_m3)),
        peak_temperature_k=float(np.max(evolved.temperature_k)),
        routing_solution=solution,
        gbp_result=after_result,
        gbp_sweeps=before_result.sweeps + after_result.sweeps,
        centralized_baseline=centralized,
        flooding_baseline=flooding,
        clairvoyant_baseline=clairvoyant,
        ember_result=ember_result,
        shortest_dose=centralized.hazard_dose,
        ember_dose=ember_result.hazard_dose,
        receiver_information_gain_nats=information_gain,
        radio_packets=total_radio.broadcasts,
        radio_bytes=total_radio.bytes_air,
        radio_stats=total_radio,
    )


__all__ = ["MissionResult", "MissionWorld", "build_mission_world", "run_mission"]
