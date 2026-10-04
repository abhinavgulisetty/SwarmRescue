"""Connected-region inference with certified fine-geometry route refinement.

This host prototype bounds the retained inference/routing arena. The fine map,
certificate construction, and refinement scratch remain host-side and are
explicitly outside the embedded memory claim.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field, replace
from itertools import pairwise

import numpy as np

from cascade.extract import TraversabilityGraph, nearest_node
from ember.gbp import gbp_storage_bytes
from ember.physical import PhysicalScan, scan_physical_hazard
from ember.physical_team import PhysicalRobotTeam
from ember.robot_state import RobotState
from ember.routing import oracle_state_cost
from ember.routing.deadline import safeguard_route_deadline
from mcu import MCUArena, MCUOutOfMemory


def _shortest(adjacent, start, goal, allowed, risk=None):
    distance = {start: 0.0}
    parent = {}
    queue = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance[node]:
            continue
        if node == goal:
            route = [node]
            while route[-1] != start:
                route.append(parent[route[-1]])
            return tuple(reversed(route)), cost
        for target, length in adjacent[node]:
            if not allowed[target]:
                continue
            step = length if risk is None else length * (1.0 + risk[target])
            candidate = cost + step
            if candidate < distance.get(target, math.inf):
                distance[target] = candidate
                parent[target] = node
                heapq.heappush(queue, (candidate, target))
    raise RuntimeError("connected-region certificate could not be refined")


@dataclass(frozen=True, slots=True)
class RegionMap:
    fine: TraversabilityGraph
    graph: TraversabilityGraph
    node_to_region: np.ndarray
    anchors: np.ndarray
    members: tuple[np.ndarray, ...]
    signatures: tuple[frozenset, ...]
    certificates: tuple[tuple[int, ...], ...]
    travel_lengths_m: np.ndarray
    adjacent: tuple[tuple[tuple[int, float], ...], ...]
    motion_adjacent: tuple[tuple[tuple[int, float], ...], ...]
    region_cells: int

    @classmethod
    def build(cls, fine, *, region_cells=4):
        if not isinstance(region_cells, int) or region_cells < 1:
            raise ValueError("region_cells must be a positive integer")
        if fine.n_nodes < 1:
            raise ValueError("cannot partition an empty graph")
        parent = np.arange(fine.n_nodes)
        blocks = fine.cells // np.array([region_cells, region_cells, 1])

        def root(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = int(parent[node])
            return node

        positions = fine.world_positions()
        adjacent = [[] for _ in range(fine.n_nodes)]
        motion_adjacent = [[] for _ in range(fine.n_nodes)]
        for (left, right), difficulty in zip(fine.edges, fine.edge_cost, strict=True):
            left, right = int(left), int(right)
            length = float(np.linalg.norm(positions[left] - positions[right]))
            if length <= 0 or not math.isfinite(length):
                raise ValueError("fine edges must have positive physical length")
            adjacent[left].append((right, length))
            adjacent[right].append((left, length))
            effort = max(length, float(difficulty) * fine.grid.cell[0])
            motion_adjacent[left].append((right, effort))
            motion_adjacent[right].append((left, effort))
            if np.array_equal(blocks[left], blocks[right]):
                a, b = root(left), root(right)
                parent[max(a, b)] = min(a, b)
        groups = {}
        for node in range(fine.n_nodes):
            groups.setdefault(root(node), []).append(node)
        # Region ordering follows stable voxel identities, not transient node IDs.
        groups = sorted(
            groups.values(), key=lambda g: min(tuple(fine.cells[n]) for n in g)
        )
        members = tuple(np.asarray(group, dtype=np.int32) for group in groups)
        node_to_region = np.empty(fine.n_nodes, dtype=np.int32)
        anchors = []
        signatures = []
        for region, nodes in enumerate(members):
            node_to_region[nodes] = region
            center = positions[nodes].mean(axis=0)
            anchor = min(
                nodes,
                key=lambda n: (
                    float(np.linalg.norm(positions[n] - center)),
                    tuple(fine.cells[n]),
                ),
            )
            anchors.append(int(anchor))
            signatures.append(
                frozenset(
                    (tuple(fine.cells[n]), float(fine.surface_z[n])) for n in nodes
                )
            )
        anchors = np.asarray(anchors, dtype=np.int32)
        pairs = sorted(
            {
                tuple(sorted((int(node_to_region[a]), int(node_to_region[b]))))
                for a, b in fine.edges
                if node_to_region[a] != node_to_region[b]
            }
        )
        certificates, lengths, efforts = [], [], []
        for a, b in pairs:
            allowed = (node_to_region == a) | (node_to_region == b)
            path, effort = _shortest(
                motion_adjacent, int(anchors[a]), int(anchors[b]), allowed
            )
            certificates.append(path)
            efforts.append(effort)
            lengths.append(
                sum(
                    float(np.linalg.norm(positions[i] - positions[j]))
                    for i, j in pairwise(path)
                )
            )
        lengths = np.asarray(lengths, dtype=float)
        graph = TraversabilityGraph(
            fine.grid,
            fine.cells[anchors].copy(),
            fine.surface_z[anchors].copy(),
            np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
            np.asarray(efforts) / fine.grid.cell[0],
        )
        return cls(
            fine,
            graph,
            node_to_region,
            anchors,
            members,
            tuple(signatures),
            tuple(certificates),
            lengths,
            tuple(tuple(sorted(neighbors)) for neighbors in adjacent),
            tuple(tuple(sorted(neighbors)) for neighbors in motion_adjacent),
            region_cells,
        )

    def aggregate(self, scan):
        """One peak observation per region; sample count never shrinks variance.

        This is a deliberately biased peak-cost representation, not an unbiased
        Gaussian estimate of every fine cell or a calibrated regional maximum.
        """
        if not scan.nodes.size:
            return scan
        selected = {}
        for index, node in enumerate(scan.nodes):
            region = int(self.node_to_region[node])
            previous = selected.get(region)
            if previous is None or scan.mean[index] > scan.mean[previous]:
                selected[region] = index
        regions = np.array(sorted(selected), dtype=np.int64)
        indices = np.array([selected[int(region)] for region in regions])
        return PhysicalScan(
            regions,
            scan.mean[indices],
            scan.variance[indices],
            scan.temperature_k[indices],
            scan.smoke_kg_m3[indices],
        )

    def corridor(self, start, goal, regional_route, risk):
        allowed = np.isin(self.node_to_region, regional_route)
        return _shortest(self.motion_adjacent, start, goal, allowed, risk)[0], allowed

    def refine(self, start, goal, regional_route, risk, available_time_s=None):
        """Preserve terrain effort; widen a time-infeasible regional corridor.

        Motion effort is a conservative uncalibrated clock proxy. Construction
        and widening are host operations, not a measured embedded kernel.
        """
        regions = set(int(region) for region in regional_route)
        adjacent = self.graph.adjacency()
        fine = self.fine
        positions = fine.world_positions()
        expansions = 0
        while True:
            nodes, allowed = self.corridor(start, goal, tuple(regions), risk)
            included = np.flatnonzero(allowed)
            if available_time_s is None:
                return nodes, None, len(included), expansions
            mapping = np.full(fine.n_nodes, -1, dtype=np.int32)
            mapping[included] = np.arange(len(included))
            edge_mask = allowed[fine.edges[:, 0]] & allowed[fine.edges[:, 1]]
            edges = fine.edges[edge_mask]
            lengths = np.linalg.norm(
                positions[edges[:, 0]] - positions[edges[:, 1]], axis=1
            )
            effort = np.maximum(lengths, fine.edge_cost[edge_mask] * fine.grid.cell[0])
            corridor = TraversabilityGraph(
                fine.grid,
                fine.cells[included],
                fine.surface_z[included],
                mapping[edges],
                effort,
            )
            constrained = safeguard_route_deadline(
                corridor,
                int(mapping[start]),
                (int(mapping[goal]),),
                risk[included],
                tuple(int(mapping[n]) for n in nodes),
                available_steps=math.floor(available_time_s / 0.25),
                walking_speed_m_s=0.30,
                dt_s=0.25,
            )
            nodes = tuple(int(included[n]) for n in constrained.route)
            if constrained.feasible:
                return nodes, constrained, len(included), expansions
            expanded = regions | {
                target for region in regions for target, _ in adjacent[region]
            }
            if expanded == regions:
                return nodes, constrained, len(included), expansions
            regions = expanded
            expansions += 1


@dataclass(slots=True)
class RegionRobotState(RobotState):
    fine_node_count: int = 0
    hazard_floor: np.ndarray = field(init=False)
    fine_mean: np.ndarray = field(init=False)
    fine_variance: np.ndarray = field(init=False)
    fine_observed: np.ndarray = field(init=False)

    def __post_init__(self):
        RobotState.__post_init__(self)
        self.hazard_floor = np.frombuffer(
            self.arena.alloc(8 * self.graph.n_nodes, alignment=8), dtype=np.float64
        )
        self.hazard_floor[:] = 0
        self.fine_mean = np.frombuffer(
            self.arena.alloc(4 * self.fine_node_count), dtype=np.float32
        )
        self.fine_variance = np.frombuffer(
            self.arena.alloc(4 * self.fine_node_count), dtype=np.float32
        )
        self.fine_observed = np.frombuffer(
            self.arena.alloc(self.fine_node_count), dtype=np.bool_
        )
        self.fine_mean[:] = 0
        self.fine_variance[:] = 1
        self.fine_observed[:] = False
        self._routing_offset = self.arena.used

    def consume(self):
        packets = tuple(self.queue)
        accepted = RobotState.consume(self)
        # A cold peer measurement must not wash out a delivered hotspot.
        for packet in packets:
            retained = np.array(
                [
                    (packet.sender, packet.sequence, int(cell)) in self.seen_factors
                    for cell in packet.cells
                ]
            )
            np.maximum.at(
                self.hazard_floor,
                packet.cells[retained],
                packet.information_q16[retained].astype(float)
                / packet.precision_q16[retained],
            )
        return accepted


class RegionRobotTeam(PhysicalRobotTeam):
    """Experimental spatial abstraction; no full posterior is shared."""

    def __init__(self, graph, positions_m, *, region_cells=4, **kwargs):
        self.regions = RegionMap.build(graph, region_cells=region_cells)
        self.refinement_peak_nodes = 0
        self.refinement_expansions = 0
        self.discarded_region_factors = 0
        super().__init__(self.regions.graph, positions_m, **kwargs)

    def _new_robot(self, robot_id, position):
        nodes, edges = self.graph.n_nodes, self.graph.n_edges
        capacity = (
            gbp_storage_bytes(nodes, edges)
            + 8 * nodes
            + 18 * nodes
            + 8 * (2 * edges + nodes)
            + 12
            + 9 * self.regions.fine.n_nodes
            + 16
            + 1024
            + (18 * 2048 if self.latest_evidence else 0)
        )
        if capacity >= 200 * 1024:
            raise MCUOutOfMemory("regional numeric workspace exceeds 200 KiB")
        return RegionRobotState(
            robot_id,
            self.graph,
            position,
            arena=MCUArena(capacity),
            factor_capacity=2048,
            drop_on_factor_overflow=True,
            latest_evidence=self.latest_evidence,
            queue_capacity=64,
            fine_node_count=self.regions.fine.n_nodes,
        )

    def _scan(self, domain, state, position, rng, *, robot=None, **kwargs):
        scan = scan_physical_hazard(
            self.regions.fine, domain, state, position, rng, **kwargs
        )
        if robot is not None:
            # Retain only direct visible readings, not simulator truth or peers'
            # posteriors. These arena-backed arrays cost nine bytes per fine cell.
            robot.fine_mean[scan.nodes] = scan.mean
            robot.fine_variance[scan.nodes] = scan.variance
            robot.fine_observed[scan.nodes] = True
        return self.regions.aggregate(scan)

    def plan(
        self,
        index,
        domain,
        state,
        start,
        goal,
        previous,
        rng,
        *,
        beta=-2.0,
        available_time_s=None,
    ):
        region_map = self.regions
        fine = region_map.fine
        positions = fine.world_positions()
        start_node = nearest_node(fine, tuple(start))
        goal_node = nearest_node(fine, tuple(goal))
        coarse_positions = self.graph.world_positions()
        start_region = int(region_map.node_to_region[start_node])
        goal_region = int(region_map.node_to_region[goal_node])
        previous_nodes = np.array([nearest_node(fine, tuple(p)) for p in previous])
        previous_regions = region_map.node_to_region[previous_nodes]
        result = super().plan(
            index,
            domain,
            state,
            coarse_positions[start_region],
            coarse_positions[goal_region],
            coarse_positions[previous_regions],
            rng,
            beta=beta,
            available_time_s=available_time_s,
            travel_edge_lengths_m=region_map.travel_lengths_m,
            risk_floor=self.robots[index].hazard_floor,
        )
        policy = self.robots[index].routing.route(start_region)
        # The deadline safeguard may replace the integer proposal. Recover its
        # actual regional route from the returned waypoints, not the old policy.
        if result.deadline_intervened:
            policy = tuple(nearest_node(self.graph, tuple(p)) for p in result.waypoints)
        risk = result.risk_cost[region_map.node_to_region]
        robot = self.robots[index]
        observed = robot.fine_observed
        risk[observed] = np.maximum(
            robot.fine_mean[observed].astype(float)
            + beta * np.sqrt(robot.fine_variance[observed].astype(float)),
            1.0e-4,
        )
        nodes, constrained, refinement_nodes, expansions = region_map.refine(
            start_node, goal_node, policy, risk, available_time_s
        )
        self.refinement_peak_nodes = max(self.refinement_peak_nodes, refinement_nodes)
        self.refinement_expansions += expansions
        waypoints = positions[list(nodes)]
        if np.linalg.norm(waypoints[0] - start) > 0.05:
            waypoints = np.vstack((start, waypoints))
        proxy = oracle_state_cost(fine, domain, state)
        return replace(
            result,
            waypoints=waypoints,
            prior_route_cost=float(risk[previous_nodes].sum()),
            posterior_route_cost=float(risk[list(nodes)].sum()),
            baseline_proxy_cost=float(proxy[previous_nodes].sum()),
            planned_proxy_cost=float(proxy[list(nodes)].sum()),
            deadline_intervened=result.deadline_intervened
            or (constrained.proposal_replaced if constrained else False),
            modeled_deadline_feasible=constrained.feasible if constrained else None,
            estimated_travel_time_s=constrained.travel_steps * 0.25
            if constrained
            else None,
        )

    def replace_graph(self, graph, *, process_variance=0.02):
        previous = self.regions
        if (
            graph.grid.origin != previous.fine.grid.origin
            or graph.grid.cell != previous.fine.grid.cell
        ):
            raise ValueError("geometry rebuild must keep the same voxel lattice")
        current = RegionMap.build(graph, region_cells=previous.region_cells)
        stable = {
            signature
            for signature in current.signatures
            if signature in previous.signatures
        }
        stable_indices = {
            index
            for index, signature in enumerate(previous.signatures)
            if signature in stable
        }
        old_floors = []
        for robot in self.robots:
            robot.consume()
            changed = np.array(
                [
                    index
                    for index in range(self.graph.n_nodes)
                    if index not in stable_indices
                ],
                dtype=np.int64,
            )
            robot.engine.local_precision[changed] = 25.0
            robot.engine.local_information[changed] = 7.5
            retained = {key for key in robot.seen_factors if key[2] in stable_indices}
            self.discarded_region_factors += len(robot.seen_factors) - len(retained)
            robot.seen_factors = retained
            old_floors.append(
                {previous.signatures[i]: robot.hazard_floor[i] for i in stable_indices}
            )
        old_robots = self.robots
        old_lookup = {
            (tuple(cell), float(height)): i
            for i, (cell, height) in enumerate(
                zip(previous.fine.cells, previous.fine.surface_z, strict=True)
            )
        }
        surviving = [
            (old_lookup[(tuple(cell), float(height))], i)
            for i, (cell, height) in enumerate(
                zip(graph.cells, graph.surface_z, strict=True)
            )
            if (tuple(cell), float(height)) in old_lookup
        ]
        self.regions = current
        super().replace_graph(current.graph, process_variance=process_variance)
        for old, robot, floors in zip(old_robots, self.robots, old_floors, strict=True):
            if surviving:
                old_nodes, new_nodes = np.array(surviving).T
                robot.fine_mean[new_nodes] = old.fine_mean[old_nodes]
                robot.fine_variance[new_nodes] = (
                    old.fine_variance[old_nodes] + process_variance
                )
                robot.fine_observed[new_nodes] = old.fine_observed[old_nodes]
            for index, signature in enumerate(current.signatures):
                robot.hazard_floor[index] = floors.get(signature, 0.0)
