"""Host integration of independent robot beliefs, physical scans and radio.

Every robot receives the selected simulator planning graph. Larger scenes
can require more than 200 KB of numeric arena memory; this is a physical host
reference, not the embedded Phase-1 acceptance gate. Transport is synchronous
within an epoch, so no old-topology packets remain in flight after a rebuild.
"""

from __future__ import annotations

from dataclasses import replace
from xml.etree import ElementTree as ET

import numpy as np

from cascade.extract import (
    GRID_MARGIN_M,
    TraversabilityGraph,
    build_graph,
    dijkstra_from,
    nearest_node,
    path_from,
    rasterize,
)
from cascade.mjcf import settled_poses, to_settled_mjcf
from cascade.robot import build_robot, place_robot
from ember.comms import PACKET_HEADER_BYTES, UPDATE_BYTES, GaussianPacket
from ember.gbp import gbp_storage_bytes
from ember.geometry_sensing import radio_wall_counts
from ember.physical import plan_belief_route, scan_physical_hazard
from ember.radio import RadioMedium, RadioStats
from ember.robot_state import RobotState
from ember.routing import LMDPConfig, build_lmdp
from ember.voi import gaussian_kl, solve_adjoint_sparse
from mcu import MCUArena


def collision_aware_params(params, spec):
    """Conservative clearance of all colliding robot parts, including its mast.

    Geom bounding spheres supply a yaw-independent swept disc. Non-colliding
    presentation beacons do not enlarge the planning footprint.
    """
    import mujoco

    body, _ = build_robot(spec, (0.0, 0.0, 0.0))
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "worldbody").append(body)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    positions = data.geom_xpos[colliding]
    bounds = model.geom_rbound[colliding]
    radius = float(np.max(np.linalg.norm(positions[:, :2], axis=1) + bounds)) + 0.01
    clearance = spec.ride_height_m + float(np.max(positions[:, 2] + bounds)) + 0.01
    return replace(
        params,
        traversability=replace(
            params.traversability,
            robot_radius_m=max(params.traversability.robot_radius_m, radius),
            min_clearance_m=max(params.traversability.min_clearance_m, clearance),
        ),
    )


def place_collision_free_team(swarm, model, data, graph, footprint):
    """Choose supported exterior starts and graph paths for every robot.

    The actual robot collision geometry, including its mast, is tested at every
    candidate. A candidate must also reach the assigned goal on the supplied map.
    """
    positions = graph.world_positions()
    depth = np.minimum.reduce(
        (
            positions[:, 0] - footprint.x0,
            footprint.x1 - positions[:, 0],
            positions[:, 1] - footprint.y0,
            footprint.y1 - positions[:, 1],
        )
    )
    candidates = np.flatnonzero((depth <= -0.35) & (np.abs(positions[:, 2]) <= 0.05))

    def penetrates(member):
        return any(
            contact.dist < -0.001
            and (
                (int(contact.geom1) in member.geom_ids)
                != (int(contact.geom2) in member.geom_ids)
            )
            for contact in data.contact[: data.ncon]
        )

    count = 0
    for index, member in enumerate(swarm.members):
        was_penetrating = penetrates(member)
        assignment = member.assignment
        goal = nearest_node(graph, tuple(assignment.route[-1]))
        distance, previous = dijkstra_from(graph, goal)
        reachable = candidates[np.isfinite(distance[candidates])]
        order = np.argsort(
            np.linalg.norm(
                positions[reachable, :2] - np.asarray(assignment.spawn[:2]),
                axis=1,
            ),
            kind="stable",
        )
        for node in reachable[order]:
            point = positions[node].copy()
            point[2] += assignment.spec.spawn_height_m
            place_robot(model, data, assignment.spec, tuple(point), assignment.yaw_rad)
            if penetrates(member):
                continue
            path = list(reversed(path_from(distance, previous, goal, int(node))))
            swarm.reroute(index, positions[path], goal_label=assignment.goal_label)
            member.assignment = replace(member.assignment, spawn=tuple(point))
            count += int(was_penetrating)
            break
        else:
            raise RuntimeError(
                f"no collision-free reachable exterior spawn for {member.name}"
            )
    return count


def ground_mission_graph(graph, *, maximum_height_m=1.0):
    """Induced ground-level planning graph for this demo's ground-level sites."""
    keep = np.flatnonzero(graph.surface_z <= maximum_height_m)
    if not keep.size:
        raise RuntimeError("the current scene has no ground-level planning cells")
    if keep.size == graph.n_nodes:
        return graph
    mapping = np.full(graph.n_nodes, -1, dtype=np.int32)
    mapping[keep] = np.arange(keep.size)
    selected_edges = np.all(mapping[graph.edges] >= 0, axis=1)
    return TraversabilityGraph(
        graph.grid,
        graph.cells[keep],
        graph.surface_z[keep],
        mapping[graph.edges[selected_edges]],
        graph.edge_cost[selected_edges],
    )


def rebuild_physical_graph(model, data, structure, params, *, margin_m=GRID_MARGIN_M):
    """Rasterize exact current structural poses in a robot-free snapshot."""
    import mujoco

    poses = settled_poses(model, data, structure, snap_to_authored=False)
    snapshot = mujoco.MjModel.from_xml_string(to_settled_mjcf(structure, poses))
    snapshot_data = mujoco.MjData(snapshot)
    mujoco.mj_forward(snapshot, snapshot_data)
    grid = rasterize(
        snapshot,
        snapshot_data,
        params.building.footprint_m,
        params.building.total_height_m,
        params.traversability,
        margin_m=margin_m,
    )
    graph = build_graph(grid, params.traversability, snapshot, snapshot_data)
    if graph.n_nodes == 0:
        raise RuntimeError("current structural geometry has no traversable cells")
    return graph


def select_source_packets(packets, cells, scores, budget_bytes):
    """Rank updates while retaining the original source/sequence/cell IDs.

    A header is charged for each touched source packet. Repacketizing the chosen
    cells with fresh IDs would let a returning relay double-count local evidence.
    """
    if not isinstance(budget_bytes, int) or budget_bytes < 0:
        raise ValueError("packet budget must be a nonnegative integer")
    priorities = dict(zip(map(int, cells), map(float, scores), strict=True))
    choices = sorted(
        (
            (-priorities[int(cell)], int(cell), index, offset)
            for index, packet in enumerate(packets)
            for offset, cell in enumerate(packet.cells)
        ),
    )
    selected = {}
    used = 0
    for negative_score, _, index, offset in choices:
        extra = UPDATE_BYTES + (0 if index in selected else PACKET_HEADER_BYTES)
        if negative_score >= 0 or used + extra > budget_bytes:
            continue
        selected.setdefault(index, []).append(offset)
        used += extra
    return tuple(
        GaussianPacket(
            packet.sender,
            packet.sequence,
            packet.ttl,
            packet.provenance,
            packet.cells[selected[index]],
            packet.information_q16[selected[index]],
            packet.precision_q16[selected[index]],
        )
        for index, packet in enumerate(packets)
        if index in selected
    )


class PhysicalRobotTeam:
    """Independent local maps with interleaved GBP and actual radio delivery."""

    def __init__(
        self,
        graph,
        positions_m,
        *,
        loss_seed,
        packet_budget_bytes=128,
        radio_config=None,
    ):
        positions = np.asarray(positions_m, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
            raise ValueError("physical team requires at least two 3D positions")
        self.graph = graph
        self.packet_budget_bytes = packet_budget_bytes
        self.radio = RadioMedium(positions, loss_seed=loss_seed, config=radio_config)
        self.robots = tuple(
            self._new_robot(i, point) for i, point in enumerate(positions)
        )
        self.radio_stats = RadioStats()
        self.geometry_revisions = 0
        self.preserved_factors = 0
        self.scans = ()
        self.max_wall_count = 0
        self.arena_high_water_bytes = 0

    def _new_robot(self, robot_id, position):
        # Numeric GBP state plus reusable integer routing buffers and alignment.
        nodes, edges = self.graph.n_nodes, self.graph.n_edges
        # build_lmdp has at most two transitions per edge plus one per node.
        routing_bytes = 18 * nodes + 8 * (2 * edges + nodes) + 12
        capacity = max(
            200 * 1024, gbp_storage_bytes(nodes, edges) + routing_bytes + 1024
        )
        return RobotState(robot_id, self.graph, position, arena=MCUArena(capacity))

    def observe_and_exchange(
        self,
        domain,
        state,
        positions_m,
        goals_m,
        rng,
        *,
        model,
        data,
        scan_radius_m=4.5,
        beta=-2.0,
        sensor_config=None,
    ):
        positions = np.asarray(positions_m, dtype=float)
        goals = np.asarray(goals_m, dtype=float)
        if (
            positions.shape != self.radio.positions_m.shape
            or goals.shape != positions.shape
        ):
            raise ValueError("one current position and goal is required per robot")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(goals)):
            raise ValueError("robot positions and goals must be finite")
        self.radio.positions_m[:] = positions + np.array([0.0, 0.0, 0.3])
        self.radio.wall_counts[:] = radio_wall_counts(
            model, data, self.radio.positions_m
        )
        self.max_wall_count = max(
            self.max_wall_count, int(self.radio.wall_counts.max())
        )
        scans, outbound = [], []
        for robot, position, goal in zip(self.robots, positions, goals, strict=True):
            robot.position_m[:] = position
            robot.consume()
            prior = robot.engine.belief()
            scan = scan_physical_hazard(
                self.graph,
                domain,
                state,
                position,
                rng,
                scan_radius_m=scan_radius_m,
                sensor_config=sensor_config,
                scene_model=model,
                scene_data=data,
            )
            scans.append(scan)
            packets = robot.observe(scan.nodes, scan.mean, scan.variance)
            robot.consume()
            if not scan.nodes.size:
                outbound.append(())
                continue
            goal_node = nearest_node(self.graph, tuple(goal))
            route_model = build_lmdp(
                self.graph,
                prior.routing_cost(beta),
                (goal_node,),
                config=LMDPConfig(
                    self_loop_weight=0.0, minimum_state_cost=0.02, max_sweeps=4000
                ),
            )
            source = np.zeros(self.graph.n_nodes)
            source[nearest_node(self.graph, tuple(position))] = 1.0
            adjoint = solve_adjoint_sparse(route_model, source)
            if not adjoint.converged:
                raise RuntimeError("physical communication adjoint did not converge")
            posterior = prior.update(scan.nodes, scan.mean, scan.variance)
            kl = gaussian_kl(
                prior.mean[scan.nodes],
                prior.variance[scan.nodes],
                posterior.mean[scan.nodes],
                posterior.variance[scan.nodes],
            )
            scores = kl * adjoint.field[scan.nodes]
            outbound.append(
                select_source_packets(
                    packets,
                    scan.nodes,
                    scores,
                    self.packet_budget_bytes,
                )
            )
        # Every robot has completed its own scan before any peer evidence arrives.
        for sender, packets in enumerate(outbound):
            for packet in packets:
                flood = self.radio.flood_packet(sender, packet, rng)
                self.radio_stats += flood.stats
                for receiver, received in enumerate(flood.received):
                    if receiver != sender:
                        for delivered in received:
                            self.robots[receiver].enqueue(delivered)
        pending = set(range(len(self.robots)))
        cap = max(robot.gbp_config.max_sweeps for robot in self.robots)
        message_count = self.robots[0].engine.source.size
        queued_cells = max(
            sum(packet.cells.size for packet in r.queue) for r in self.robots
        )
        available = min(r.cycles_per_tick for r in self.robots) - (
            12 * self.graph.n_nodes + 16 * message_count + 40 * queued_cells
        )
        block_size = available // 64
        if block_size < 1:
            raise RuntimeError(
                "physical inference preparation exceeds the modeled tick budget"
            )
        blocks = []
        for start in range(0, message_count, block_size):
            mask = np.zeros(message_count, dtype=bool)
            mask[start : start + block_size] = True
            blocks.append(mask)
        if not blocks:
            blocks = [np.zeros(0, dtype=bool)]
        for _ in range(cap):
            residuals = {index: 0.0 for index in pending}
            for block in blocks:
                for index in sorted(pending):
                    robot = self.robots[index]
                    if not robot.tick(block):
                        raise RuntimeError(
                            "physical inference block exceeded its tick budget"
                        )
                    residuals[index] = max(residuals[index], robot.last_residual)
            for index in sorted(pending):
                if residuals[index] <= self.robots[index].gbp_config.tolerance:
                    pending.remove(index)
            if not pending:
                break
        if pending:
            raise RuntimeError("physical robot inference exceeded its sweep budget")
        self.scans = tuple(scans)

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
        result = plan_belief_route(
            self.graph,
            domain,
            state,
            start,
            goal,
            previous,
            rng,
            robot_state=self.robots[index],
            local_scan=self.scans[index],
            ingest_scan=False,
            infer_belief=False,
            native_routing=True,
            available_time_s=available_time_s,
            beta=beta,
        )
        self.arena_high_water_bytes = max(
            self.arena_high_water_bytes,
            self.robots[index].arena.high_water,
        )
        return result

    def replace_graph(self, graph, *, process_variance=0.02):
        """Transfer local factors on surviving cells, then rebuild spatial factors.

        Posterior marginals are never inserted as fresh independent evidence.
        Removed cells lose their factors. Sequence epochs continue across maps.
        """
        if not np.isfinite(process_variance) or process_variance < 0:
            raise ValueError("process variance must be finite and nonnegative")
        if (
            graph.grid.origin != self.graph.grid.origin
            or graph.grid.cell != self.graph.grid.cell
        ):
            raise ValueError("geometry rebuild must keep the same voxel lattice")
        lookup = {tuple(cell): i for i, cell in enumerate(graph.cells)}
        mapping = {}
        tolerance = 0.5 * graph.grid.cell[2]
        for old, cell in enumerate(self.graph.cells):
            new = lookup.get(tuple(cell))
            if (
                new is not None
                and abs(self.graph.surface_z[old] - graph.surface_z[new]) <= tolerance
            ):
                mapping[old] = new
        old_robots = self.robots
        self.graph = graph
        self.robots = tuple(
            self._new_robot(r.robot_id, r.position_m) for r in old_robots
        )
        old_indices = np.array(list(mapping), dtype=int)
        new_indices = np.array(list(mapping.values()), dtype=int)
        for old, new in zip(old_robots, self.robots, strict=True):
            old.consume()
            old.engine.age(old_indices, process_variance)
            new.engine.local_precision[new_indices] = old.engine.local_precision[
                old_indices
            ]
            new.engine.local_information[new_indices] = old.engine.local_information[
                old_indices
            ]
            new.seen_factors = {
                (sender, sequence, mapping[cell])
                for sender, sequence, cell in old.seen_factors
                if cell in mapping
            }
            self.preserved_factors += len(new.seen_factors)
            new.next_sequence = old.next_sequence
            new.sensor_state.update(old.sensor_state)
            new.deadline_misses = old.deadline_misses
            new.inference_sweeps = old.inference_sweeps
            self.arena_high_water_bytes = max(
                self.arena_high_water_bytes, old.arena.high_water
            )
        self.geometry_revisions += 1
        self.scans = ()
        return mapping
