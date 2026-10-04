"""Bridge noisy local hazard sensing to a physically drivable route."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np

from cascade.extract import TraversabilityGraph, nearest_node
from ember.belief import (
    GaussianBelief,
    HazardSensorConfig,
    observe_state_cost,
    sample_hazard,
)
from ember.geometry_sensing import box_raycasts
from ember.robot_state import RobotState
from ember.routing import (
    LMDPConfig,
    build_lmdp,
    greedy_route,
    oracle_state_cost,
    solve_jacobi,
)
from ember.routing.deadline import safeguard_route_deadline
from hazard import HazardDomain, HazardState


@dataclass(frozen=True, slots=True)
class PhysicalScan:
    """One noisy, visibility-filtered sensor epoch, reusable without resampling."""

    nodes: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    temperature_k: np.ndarray
    smoke_kg_m3: np.ndarray


def scan_physical_hazard(
    graph,
    domain,
    state,
    start_m,
    rng,
    *,
    scan_radius_m=4.5,
    sensor_config=None,
    visible_nodes=None,
    scene_model=None,
    scene_data=None,
):
    start = np.asarray(start_m, dtype=float)
    if start.shape != (3,) or not np.all(np.isfinite(start)):
        raise ValueError("scan origin must be a finite 3D point")
    if scan_radius_m <= 0 or not math.isfinite(scan_radius_m):
        raise ValueError("scan radius must be finite and positive")
    positions = graph.world_positions()
    inside = np.all(
        (positions >= np.asarray(domain.origin_m))
        & (positions <= np.asarray(domain.bounds_max_m)),
        axis=1,
    )
    observed = np.flatnonzero(
        inside & (np.linalg.norm(positions - start, axis=1) <= scan_radius_m)
    )
    if (scene_model is None) != (scene_data is None):
        raise ValueError("physical scene requires both model and data")
    if scene_model is not None:
        offset = np.array([0.0, 0.0, 0.3])
        hits, _ = box_raycasts(
            scene_model, scene_data, start + offset, positions[observed] + offset
        )
        distances = np.linalg.norm(positions[observed] - start, axis=1)
        observed = observed[hits >= distances - 1e-6]
    if visible_nodes is not None:
        visible = np.asarray(visible_nodes, dtype=bool)
        if visible.shape != (graph.n_nodes,):
            raise ValueError("visible_nodes must contain one flag per graph node")
        observed = observed[visible[observed]]
    if not observed.size:
        empty = np.zeros(0)
        return PhysicalScan(observed, empty, empty, empty, empty)
    reading = sample_hazard(state, domain, positions[observed], rng, sensor_config)
    mean, variance = observe_state_cost(reading)
    return PhysicalScan(
        observed, mean, variance, reading.temperature_k, reading.smoke_kg_m3
    )


@dataclass(frozen=True, slots=True)
class PhysicalRouteResult:
    """Evidence retained when a belief update replaces a physical route."""

    waypoints: np.ndarray
    observed_nodes: int
    peak_temperature_k: float
    peak_smoke_kg_m3: float
    prior_route_cost: float
    posterior_route_cost: float
    baseline_proxy_cost: float
    planned_proxy_cost: float
    sweeps: int
    residual: float
    converged: bool
    observed_node_indices: np.ndarray
    belief_mean: np.ndarray
    belief_std: np.ndarray
    risk_cost: np.ndarray
    beta: float
    persistent_robot_belief: bool = False
    routing_arithmetic: str = "float64"
    graph_positions: np.ndarray | None = None
    deadline_intervened: bool = False
    modeled_deadline_feasible: bool | None = None
    estimated_travel_time_s: float | None = None

    @property
    def improved(self) -> bool:
        return self.posterior_route_cost < self.prior_route_cost

    @property
    def proxy_improved(self) -> bool:
        """Whether the route also improves the common observer-only proxy."""

        return self.planned_proxy_cost < self.baseline_proxy_cost


def hazard_route_change_allowed(
    previous, proposed, positions, risk, *, min_gain=0.15, deadline_intervened=False
):
    """Host route-switch inertia using the same private field for both paths.

    Integrate piecewise nearest-cell risk at <=0.125 m spacing so waypoint
    density does not decide the comparison. This is a modeled path cost, not
    observer truth or a calibrated exposure/time bound. Geometry invalidation
    bypasses this guard. A shortening deadline intervention also bypasses it.
    """
    from scipy.spatial import cKDTree

    if not math.isfinite(min_gain) or not 0 <= min_gain < 1:
        raise ValueError("route-switch gain must be finite in [0, 1)")
    if min_gain == 0:
        return True
    positions, risk = np.asarray(positions), np.asarray(risk)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or risk.shape != (len(positions),)
        or not len(positions)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(risk))
        or np.any(risk < 0)
    ):
        raise ValueError("route-switch field must have finite positions and risk")
    tree = cKDTree(positions)

    def cost(path):
        path = np.asarray(path, dtype=float)
        if (
            path.ndim != 2
            or path.shape[1] != 3
            or not len(path)
            or not np.all(np.isfinite(path))
        ):
            raise ValueError("route-switch path must contain finite 3D waypoints")
        distance, integrated = 0.0, 0.0
        for start, end in pairwise(path):
            length = float(np.linalg.norm(end - start))
            if length == 0:
                continue
            count = max(1, math.ceil(length / 0.125))
            samples = start + ((np.arange(count) + 0.5) / count)[:, None] * (
                end - start
            )
            _, nodes = tree.query(samples)
            integrated += length * float(np.mean(risk[nodes]))
            distance += length
        return integrated, distance

    old_cost, old_length = cost(previous)
    new_cost, new_length = cost(proposed)
    if deadline_intervened and new_length < old_length:
        return True
    return old_cost > 0 and new_cost <= (1 - min_gain) * old_cost


def plan_belief_route(
    graph: TraversabilityGraph,
    domain: HazardDomain,
    state: HazardState,
    start_m: np.ndarray,
    goal_m: np.ndarray,
    previous_waypoints: np.ndarray,
    rng: np.random.Generator,
    *,
    scan_radius_m: float = 4.5,
    beta: float = -2.0,
    prior_mean: float = 0.30,
    prior_variance: float = 0.04,
    sensor_config: HazardSensorConfig | None = None,
    lmdp_config: LMDPConfig | None = None,
    robot_state: RobotState | None = None,
    visible_nodes: np.ndarray | None = None,
    scene_model=None,
    scene_data=None,
    local_scan: PhysicalScan | None = None,
    ingest_scan: bool = True,
    infer_belief: bool = True,
    native_routing: bool = False,
    available_time_s: float | None = None,
    planning_speed_m_s: float = 0.30,
    planning_dt_s: float = 0.25,
    travel_edge_lengths_m: np.ndarray | None = None,
    risk_floor: np.ndarray | None = None,
) -> PhysicalRouteResult:
    """Scan locally, update a Gaussian belief, and solve a new LMDP route."""

    start = np.asarray(start_m, dtype=np.float64)
    goal = np.asarray(goal_m, dtype=np.float64)
    previous = np.asarray(previous_waypoints, dtype=np.float64)
    if start.shape != (3,) or goal.shape != (3,):
        raise ValueError("start_m and goal_m must be three-dimensional points")
    if previous.ndim != 2 or previous.shape[1:] != (3,) or len(previous) == 0:
        raise ValueError("previous_waypoints must have shape (N, 3), N > 0")
    if not np.all(np.isfinite(np.vstack((start, goal, previous)))):
        raise ValueError("route points must be finite")
    if scan_radius_m <= 0.0 or not math.isfinite(scan_radius_m):
        raise ValueError("scan_radius_m must be finite and positive")

    positions = graph.world_positions()
    scan = local_scan or scan_physical_hazard(
        graph,
        domain,
        state,
        start,
        rng,
        scan_radius_m=scan_radius_m,
        sensor_config=sensor_config,
        visible_nodes=visible_nodes,
        scene_model=scene_model,
        scene_data=scene_data,
    )
    observed = scan.nodes
    if observed.size == 0 and robot_state is None:
        raise RuntimeError("the local scan contains no traversability nodes")
    if not ingest_scan and (robot_state is None or local_scan is None):
        raise ValueError("a pre-ingested scan requires persistent robot state")
    measured_mean, measured_variance = scan.mean, scan.variance
    if robot_state is None:
        belief = GaussianBelief.uniform(
            graph.n_nodes,
            mean=prior_mean,
            variance=prior_variance,
        ).update(observed, measured_mean, measured_variance)
    else:
        if robot_state.graph is not graph:
            raise ValueError(
                "robot state must be rebuilt for the current geometry graph"
            )
        robot_state.position_m[:] = start
        if ingest_scan:
            robot_state.observe(observed, measured_mean, measured_variance)
        if infer_belief:
            for _ in range(robot_state.gbp_config.max_sweeps):
                if robot_state.tick() and (
                    robot_state.last_residual <= robot_state.gbp_config.tolerance
                ):
                    break
            else:
                raise RuntimeError("physical robot inference exceeded its sweep budget")
        elif ingest_scan or robot_state.queue:
            raise ValueError(
                "precomputed inference requires fully consumed sensor evidence"
            )
        belief = robot_state.engine.belief()
    state_cost = belief.routing_cost(beta)
    if risk_floor is not None:
        floor = np.asarray(risk_floor, dtype=float)
        if floor.shape != state_cost.shape or not np.all(np.isfinite(floor)):
            raise ValueError("risk floor must be finite with one value per node")
        if np.any(floor < 0):
            raise ValueError("risk floor must be nonnegative")
        state_cost = np.maximum(state_cost, floor)

    start_node = nearest_node(graph, tuple(start))
    goal_node = nearest_node(graph, tuple(goal))
    config = lmdp_config or LMDPConfig(
        self_loop_weight=0.0,
        minimum_state_cost=0.02,
        tolerance=1.0e-5,
        max_sweeps=4_000,
    )
    model = build_lmdp(graph, state_cost, (goal_node,), config=config)
    if robot_state is None:
        solution = solve_jacobi(model)
        route_nodes = greedy_route(model, solution, start_node)
        sweeps, residual, converged = (
            solution.sweeps,
            solution.residual,
            solution.converged,
        )
    else:
        integer_solution = robot_state.plan(model, native=native_routing)
        route_nodes = list(integer_solution.route(start_node))
        sweeps = integer_solution.sweeps
        converged = integer_solution.fixed_point
        # Report the actual integer Bellman residual, including capped solves.
        from mcu.log_lmdp import log_q16_sweep

        current = integer_solution.negative_log_desirability
        following = np.empty_like(current)
        log_q16_sweep(integer_solution.model, current, following)
        residual = (
            float(np.max(np.abs(following.astype(np.int64) - current.astype(np.int64))))
            / 65536
        )
    constrained = None
    if available_time_s is not None:
        if min(planning_speed_m_s, planning_dt_s) <= 0 or not all(
            map(math.isfinite, (planning_speed_m_s, planning_dt_s))
        ):
            raise ValueError(
                "physical planning speed and time step must be positive and finite"
            )
        if not math.isfinite(available_time_s) or available_time_s < 0:
            raise ValueError(
                "remaining physical mission time must be finite and nonnegative"
            )
        # Physical graph costs are flat-cell equivalents, not distances in metres.
        distances = np.linalg.norm(
            positions[graph.edges[:, 0]] - positions[graph.edges[:, 1]],
            axis=1,
        )
        if travel_edge_lengths_m is not None:
            distances = np.asarray(travel_edge_lengths_m, dtype=float)
            if (
                distances.shape != (graph.n_edges,)
                or not np.all(np.isfinite(distances))
                or np.any(distances <= 0)
            ):
                raise ValueError("travel edge lengths must be finite and positive")
        metric_graph = replace(graph, edge_cost=distances)
        constrained = safeguard_route_deadline(
            metric_graph,
            start_node,
            (goal_node,),
            state_cost,
            route_nodes,
            available_steps=math.floor(available_time_s / planning_dt_s),
            walking_speed_m_s=planning_speed_m_s,
            dt_s=planning_dt_s,
            arena=robot_state.arena if robot_state is not None else None,
        )
        route_nodes = list(constrained.route)
    if not route_nodes:
        raise RuntimeError("the belief-aware LMDP could not reach the physical goal")

    route = positions[route_nodes]
    if np.linalg.norm(route[0] - start) > 0.05:
        route = np.vstack((start, route))
    previous_nodes = np.asarray(
        [nearest_node(graph, tuple(point)) for point in previous],
        dtype=np.int64,
    )
    # Evaluation is deliberately separate from the agent's beta-dependent
    # objective. Both routes are scored against the same simulator-truth
    # heat/smoke proxy, so search cost cannot be presented as survivor risk.
    proxy_cost = oracle_state_cost(graph, domain, state)
    return PhysicalRouteResult(
        waypoints=route,
        observed_nodes=int(observed.size),
        peak_temperature_k=float(np.max(scan.temperature_k, initial=0.0)),
        peak_smoke_kg_m3=float(np.max(scan.smoke_kg_m3, initial=0.0)),
        prior_route_cost=float(np.sum(state_cost[previous_nodes])),
        posterior_route_cost=float(np.sum(state_cost[route_nodes])),
        baseline_proxy_cost=float(np.sum(proxy_cost[previous_nodes])),
        planned_proxy_cost=float(np.sum(proxy_cost[route_nodes])),
        sweeps=sweeps,
        residual=residual,
        converged=converged,
        observed_node_indices=observed,
        belief_mean=belief.mean,
        belief_std=np.sqrt(belief.variance),
        risk_cost=state_cost,
        beta=beta,
        persistent_robot_belief=robot_state is not None,
        routing_arithmetic="integer_log_q16" if robot_state is not None else "float64",
        graph_positions=positions,
        deadline_intervened=constrained.proposal_replaced if constrained else False,
        modeled_deadline_feasible=constrained.feasible if constrained else None,
        estimated_travel_time_s=constrained.travel_steps * planning_dt_s
        if constrained
        else None,
    )
