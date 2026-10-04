"""Plain-Jacobi linearly-solvable routing for the Phase 1 oracle.

The model implements

``z(x) = exp(-q(x) / lambda) * sum_y p(y | x) z(y)``

with egress states pinned to one and absorbing death states pinned to zero.
Every nonterminal cost is strictly positive, so the Jacobi operator is a
contraction.  This float64 implementation is the transparent reference that
the later fixed-point MCU kernel and safeguarded NCA must match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cascade.extract import TraversabilityGraph
    from hazard import HazardDomain, HazardState


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _readonly(values: np.ndarray, dtype: np.dtype[np.generic]) -> np.ndarray:
    out = np.array(values, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class LMDPConfig:
    """Numerics and passive-dynamics parameters for the classical solver."""

    control_temperature: float = 1.0
    passive_cost_scale: float = 1.0
    self_loop_weight: float = 0.05
    minimum_state_cost: float = 1.0e-4
    tolerance: float = 1.0e-10
    max_sweeps: int = 10_000

    def __post_init__(self) -> None:
        _require(
            self.control_temperature > 0.0
            and math.isfinite(self.control_temperature),
            "control_temperature must be finite and positive",
        )
        _require(
            self.passive_cost_scale >= 0.0
            and math.isfinite(self.passive_cost_scale),
            "passive_cost_scale must be finite and nonnegative",
        )
        _require(
            self.self_loop_weight >= 0.0 and math.isfinite(self.self_loop_weight),
            "self_loop_weight must be finite and nonnegative",
        )
        _require(
            self.minimum_state_cost > 0.0
            and math.isfinite(self.minimum_state_cost),
            "minimum_state_cost must be finite and positive",
        )
        _require(
            math.exp(-self.minimum_state_cost / self.control_temperature) < 1.0,
            "minimum_state_cost/control_temperature is too small for a "
            "float64 contraction",
        )
        _require(
            self.tolerance > 0.0 and math.isfinite(self.tolerance),
            "tolerance must be finite and positive",
        )
        _require(
            isinstance(self.max_sweeps, int)
            and not isinstance(self.max_sweeps, bool)
            and self.max_sweeps >= 1,
            "max_sweeps must be a positive integer",
        )


@dataclass(frozen=True, slots=True)
class OracleCostConfig:
    """Dimensionless hazard proxy used before ISO tenability lands.

    This is intentionally labelled an oracle proxy rather than FED.  It makes
    the first end-to-end routing dilemma executable without claiming a fire
    safety calibration that has not yet been implemented.
    """

    base_step_cost: float = 0.02
    ambient_temperature_k: float = 293.15
    heat_onset_k: float = 313.15
    heat_scale_k: float = 100.0
    smoke_scale_kg_m3: float = 0.10
    heat_weight: float = 1.0
    smoke_weight: float = 1.0

    def __post_init__(self) -> None:
        _require(
            self.base_step_cost > 0.0 and math.isfinite(self.base_step_cost),
            "base_step_cost must be finite and positive",
        )
        _require(
            self.ambient_temperature_k > 0.0
            and math.isfinite(self.ambient_temperature_k),
            "ambient_temperature_k must be finite and positive",
        )
        _require(
            self.heat_onset_k >= self.ambient_temperature_k
            and math.isfinite(self.heat_onset_k),
            "heat_onset_k must be finite and no cooler than ambient",
        )
        _require(
            self.heat_scale_k > 0.0 and math.isfinite(self.heat_scale_k),
            "heat_scale_k must be finite and positive",
        )
        _require(
            self.smoke_scale_kg_m3 > 0.0
            and math.isfinite(self.smoke_scale_kg_m3),
            "smoke_scale_kg_m3 must be finite and positive",
        )
        _require(
            self.heat_weight >= 0.0 and math.isfinite(self.heat_weight),
            "heat_weight must be finite and nonnegative",
        )
        _require(
            self.smoke_weight >= 0.0 and math.isfinite(self.smoke_weight),
            "smoke_weight must be finite and nonnegative",
        )


@dataclass(frozen=True, slots=True)
class LMDPModel:
    """Sparse passive dynamics and pinned boundary states in CSR form."""

    row_ptr: np.ndarray
    col_idx: np.ndarray
    passive_prob: np.ndarray
    state_cost: np.ndarray
    exit_mask: np.ndarray
    death_mask: np.ndarray
    config: LMDPConfig

    def __post_init__(self) -> None:
        row_ptr = _readonly(self.row_ptr, np.dtype(np.int64))
        col_idx = _readonly(self.col_idx, np.dtype(np.int64))
        passive = _readonly(self.passive_prob, np.dtype(np.float64))
        cost = _readonly(self.state_cost, np.dtype(np.float64))
        exits = _readonly(self.exit_mask, np.dtype(np.bool_))
        deaths = _readonly(self.death_mask, np.dtype(np.bool_))
        n = cost.size

        _require(cost.ndim == 1 and n > 0, "state_cost must be a nonempty vector")
        _require(row_ptr.shape == (n + 1,), "row_ptr must have length N + 1")
        _require(row_ptr[0] == 0, "row_ptr must start at zero")
        _require(
            np.all(np.diff(row_ptr) > 0),
            "every passive-dynamics row must contain a transition",
        )
        _require(
            row_ptr[-1] == col_idx.size == passive.size,
            "CSR arrays have inconsistent lengths",
        )
        _require(
            exits.shape == (n,) and deaths.shape == (n,),
            "terminal masks must have length N",
        )
        _require(np.any(exits), "an LMDP needs at least one egress state")
        _require(not np.any(exits & deaths), "egress and death masks must be disjoint")
        _require(
            np.all((col_idx >= 0) & (col_idx < n)),
            "passive transition points outside the state space",
        )
        _require(
            np.all(np.isfinite(passive)) and np.all(passive >= 0.0),
            "passive probabilities must be finite and nonnegative",
        )
        _require(
            np.all(np.isfinite(cost)) and np.all(cost >= 0.0),
            "state costs must be finite and nonnegative",
        )
        nonterminal = ~(exits | deaths)
        _require(
            np.all(cost[nonterminal] >= self.config.minimum_state_cost),
            "nonterminal costs must satisfy minimum_state_cost",
        )
        row_sum = np.add.reduceat(passive, row_ptr[:-1])
        _require(
            np.allclose(row_sum, 1.0, rtol=0.0, atol=1.0e-12),
            "passive transition rows must sum to one",
        )

        object.__setattr__(self, "row_ptr", row_ptr)
        object.__setattr__(self, "col_idx", col_idx)
        object.__setattr__(self, "passive_prob", passive)
        object.__setattr__(self, "state_cost", cost)
        object.__setattr__(self, "exit_mask", exits)
        object.__setattr__(self, "death_mask", deaths)

    @property
    def n_states(self) -> int:
        return int(self.state_cost.size)

    @property
    def contraction_bound(self) -> float:
        nonterminal = ~(self.exit_mask | self.death_mask)
        if not np.any(nonterminal):
            return 0.0
        attenuation = np.exp(
            -self.state_cost[nonterminal] / self.config.control_temperature
        )
        return float(np.max(attenuation))

    def row(self, state: int) -> slice:
        if not 0 <= state < self.n_states:
            raise IndexError(f"state {state} outside [0, {self.n_states})")
        return slice(int(self.row_ptr[state]), int(self.row_ptr[state + 1]))


@dataclass(frozen=True, slots=True)
class LMDPSolution:
    """A converged desirability field and its local stochastic policy."""

    desirability: np.ndarray
    value: np.ndarray
    policy_prob: np.ndarray
    sweeps: int
    residual: float
    relative_residual: float
    converged: bool

    def __post_init__(self) -> None:
        desirability = _readonly(self.desirability, np.dtype(np.float64))
        value = _readonly(self.value, np.dtype(np.float64))
        policy = _readonly(self.policy_prob, np.dtype(np.float64))
        _require(desirability.ndim == 1, "desirability must be a vector")
        _require(value.shape == desirability.shape, "value shape must match z")
        _require(policy.ndim == 1, "policy_prob must be a vector")
        _require(self.sweeps >= 0, "sweeps cannot be negative")
        _require(self.residual >= 0.0 and math.isfinite(self.residual), "bad residual")
        _require(
            self.relative_residual >= 0.0
            and math.isfinite(self.relative_residual),
            "bad relative_residual",
        )
        _require(
            np.all(np.isfinite(desirability))
            and np.all(
                (desirability >= 0.0) & (desirability <= 1.0 + 1.0e-12)
            ),
            "desirability must be finite and in [0, 1]",
        )
        _require(
            np.all(np.isfinite(policy)) and np.all(policy >= 0.0),
            "policy probabilities must be finite and nonnegative",
        )
        object.__setattr__(self, "desirability", desirability)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "policy_prob", policy)


def _node_mask(n_states: int, nodes: tuple[int, ...] | list[int]) -> np.ndarray:
    mask = np.zeros(n_states, dtype=bool)
    for node in nodes:
        _require(
            isinstance(node, (int, np.integer)) and not isinstance(node, bool),
            "terminal nodes must be integer indices",
        )
        node = int(node)
        _require(0 <= node < n_states, f"terminal node {node} is outside the graph")
        mask[node] = True
    return mask


def build_lmdp(
    graph: TraversabilityGraph,
    state_cost: np.ndarray,
    exit_nodes: tuple[int, ...] | list[int],
    *,
    death_nodes: tuple[int, ...] | list[int] = (),
    config: LMDPConfig | None = None,
) -> LMDPModel:
    """Construct difficulty-biased passive dynamics from a traversability graph."""

    cfg = config or LMDPConfig()
    n = graph.n_nodes
    _require(n > 0, "cannot route on an empty graph")
    costs = np.asarray(state_cost, dtype=np.float64)
    _require(costs.shape == (n,), "state_cost must have one value per graph node")
    _require(
        np.all(np.isfinite(costs)) and np.all(costs >= 0.0),
        "state_cost must be finite and nonnegative",
    )
    exits = _node_mask(n, exit_nodes)
    deaths = _node_mask(n, death_nodes)
    _require(np.any(exits), "exit_nodes cannot be empty")
    _require(not np.any(exits & deaths), "exit_nodes and death_nodes must be disjoint")

    edges = np.asarray(graph.edges)
    edge_cost = np.asarray(graph.edge_cost, dtype=np.float64)
    _require(edges.ndim == 2 and edges.shape[1] == 2, "graph edges need shape (M, 2)")
    _require(np.issubdtype(edges.dtype, np.integer), "graph edges must be integers")
    _require(edge_cost.shape == (edges.shape[0],), "graph edge arrays disagree")
    _require(
        np.all(np.isfinite(edge_cost)) and np.all(edge_cost >= 0.0),
        "edge difficulty must be finite and nonnegative",
    )

    adjacency: list[dict[int, float]] = [dict() for _ in range(n)]
    for (left_raw, right_raw), difficulty in zip(edges, edge_cost, strict=True):
        left, right = int(left_raw), int(right_raw)
        _require(0 <= left < n and 0 <= right < n, "graph edge endpoint is invalid")
        old = adjacency[left].get(right, math.inf)
        adjacency[left][right] = min(old, float(difficulty))
        old = adjacency[right].get(left, math.inf)
        adjacency[right][left] = min(old, float(difficulty))

    row_ptr = [0]
    columns: list[int] = []
    probabilities: list[float] = []
    for state, neighbours in enumerate(adjacency):
        if exits[state] or deaths[state]:
            items = [(state, 1.0)]
        else:
            items = sorted(neighbours.items())
            if items:
                least_difficult = min(difficulty for _, difficulty in items)
                weighted = [
                    (
                        node,
                        math.exp(
                            -cfg.passive_cost_scale
                            * (difficulty - least_difficult)
                        ),
                    )
                    for node, difficulty in items
                ]
            else:
                weighted = []
            if cfg.self_loop_weight > 0.0:
                weighted.append((state, cfg.self_loop_weight))
            if not weighted:
                weighted = [(state, 1.0)]
            combined: dict[int, float] = {}
            for node, weight in weighted:
                combined[node] = combined.get(node, 0.0) + weight
            total = sum(combined.values())
            items = [
                (node, weight / total)
                for node, weight in sorted(combined.items())
            ]

        for node, probability in items:
            columns.append(node)
            probabilities.append(probability)
        row_ptr.append(len(columns))

    clipped_cost = np.array(costs, copy=True)
    nonterminal = ~(exits | deaths)
    clipped_cost[nonterminal] = np.maximum(
        clipped_cost[nonterminal], cfg.minimum_state_cost
    )
    clipped_cost[exits | deaths] = 0.0
    return LMDPModel(
        row_ptr=np.asarray(row_ptr, dtype=np.int64),
        col_idx=np.asarray(columns, dtype=np.int64),
        passive_prob=np.asarray(probabilities, dtype=np.float64),
        state_cost=clipped_cost,
        exit_mask=exits,
        death_mask=deaths,
        config=cfg,
    )


def _passive_matvec(model: LMDPModel, values: np.ndarray) -> np.ndarray:
    weighted = model.passive_prob * values[model.col_idx]
    return np.add.reduceat(weighted, model.row_ptr[:-1])


def _states_reaching_exit(model: LMDPModel) -> np.ndarray:
    """Numerical reachability through positive-probability reverse edges."""

    reverse: list[list[int]] = [[] for _ in range(model.n_states)]
    for source in range(model.n_states):
        if model.death_mask[source]:
            continue
        row = model.row(source)
        for target, probability in zip(
            model.col_idx[row], model.passive_prob[row], strict=True
        ):
            if probability > 0.0:
                reverse[int(target)].append(source)

    reachable = np.array(model.exit_mask, copy=True)
    stack = np.flatnonzero(model.exit_mask).tolist()
    while stack:
        target = stack.pop()
        for source in reverse[target]:
            if not reachable[source]:
                reachable[source] = True
                stack.append(source)
    return reachable


def _scaled_residual(
    candidate: np.ndarray,
    current: np.ndarray,
    relevant: np.ndarray,
) -> float:
    if not np.any(relevant):
        return 0.0
    scale = np.maximum(
        np.maximum(np.abs(candidate[relevant]), np.abs(current[relevant])),
        np.finfo(np.float64).tiny,
    )
    return float(np.max(np.abs(candidate[relevant] - current[relevant]) / scale))


def solve_jacobi(model: LMDPModel) -> LMDPSolution:
    """Relax the desirability fixed point from the safe lower bound ``z=0``."""

    cfg = model.config
    attenuation = np.exp(-model.state_cost / cfg.control_temperature)
    desirability = np.zeros(model.n_states, dtype=np.float64)
    desirability[model.exit_mask] = 1.0
    residual = math.inf
    relative_residual = math.inf
    converged = False
    sweeps = 0
    reachable = _states_reaching_exit(model)
    relevant = reachable & ~(model.exit_mask | model.death_mask)

    for sweep in range(1, cfg.max_sweeps + 1):
        sweeps = sweep
        next_desirability = attenuation * _passive_matvec(model, desirability)
        next_desirability[model.exit_mask] = 1.0
        next_desirability[model.death_mask] = 0.0
        residual = float(np.max(np.abs(next_desirability - desirability)))
        relative_residual = _scaled_residual(
            next_desirability, desirability, relevant
        )
        desirability = next_desirability
        wave_reached = np.all(desirability[reachable] > 0.0)
        if (
            wave_reached
            and residual <= cfg.tolerance
            and relative_residual <= cfg.tolerance
        ):
            converged = True
            break

    # Report the actual Bellman residual of the returned iterate too.  This is
    # the acceptance quantity the safeguarded NCA will use in Phase 2.
    fixed_point = attenuation * _passive_matvec(model, desirability)
    fixed_point[model.exit_mask] = 1.0
    fixed_point[model.death_mask] = 0.0
    residual = float(np.max(np.abs(fixed_point - desirability)))
    relative_residual = _scaled_residual(fixed_point, desirability, relevant)
    wave_reached = np.all(desirability[reachable] > 0.0)
    converged = (
        wave_reached
        and residual <= cfg.tolerance
        and relative_residual <= cfg.tolerance
    )

    value = np.full(model.n_states, np.inf, dtype=np.float64)
    positive = desirability > 0.0
    value[positive] = -cfg.control_temperature * np.log(desirability[positive])
    value[model.exit_mask] = 0.0

    numerator = model.passive_prob * desirability[model.col_idx]
    denominator = np.add.reduceat(numerator, model.row_ptr[:-1])
    policy = np.zeros_like(numerator)
    for state in range(model.n_states):
        row = model.row(state)
        if model.death_mask[state] or denominator[state] <= 0.0:
            continue
        policy[row] = numerator[row] / denominator[state]

    return LMDPSolution(
        desirability=desirability,
        value=value,
        policy_prob=policy,
        sweeps=sweeps,
        residual=residual,
        relative_residual=relative_residual,
        converged=converged,
    )


def greedy_route(
    model: LMDPModel,
    solution: LMDPSolution,
    start: int,

    *,
    max_steps: int | None = None,
) -> list[int]:
    """Extract a deterministic evaluation route from the stochastic policy.

    Self-loops and moves that do not increase desirability are ignored.  The
    strict positive-cost fixed point guarantees an increasing neighbour for
    every reachable nonterminal state; enforcing that property prevents a
    deterministic argmax from bouncing along a high-probability passive edge.
    Returns an empty list when the start has zero desirability, reaches death,
    or fails to reach egress inside the step cap.
    """

    _require(
        isinstance(start, (int, np.integer)) and not isinstance(start, bool),
        "start must be an integer state index",
    )
    start = int(start)
    _require(0 <= start < model.n_states, "start lies outside the state space")
    _require(
        solution.desirability.shape == (model.n_states,)
        and solution.policy_prob.shape == model.passive_prob.shape,
        "solution does not belong to this model",
    )
    if model.exit_mask[start]:
        return [start]
    if model.death_mask[start] or solution.desirability[start] <= 0.0:
        return []

    cap = max_steps if max_steps is not None else 4 * model.n_states
    _require(
        isinstance(cap, int) and not isinstance(cap, bool) and cap >= 1,
        "max_steps must be a positive integer",
    )
    route = [start]
    seen = {start}
    current = start
    for _ in range(cap):
        row = model.row(current)
        columns = model.col_idx[row]
        probability = solution.policy_prob[row]
        progressing = (
            (columns != current)
            & (probability > 0.0)
            & (solution.desirability[columns] > solution.desirability[current])
        )
        if not np.any(progressing):
            return []
        choices = columns[progressing]
        weights = probability[progressing]
        order = np.lexsort((choices, -weights))
        following = int(choices[int(order[0])])
        route.append(following)
        if model.exit_mask[following]:
            return route
        if model.death_mask[following] or following in seen:
            return []
        seen.add(following)
        current = following
    return []


def oracle_state_cost(
    graph: TraversabilityGraph,
    domain: HazardDomain,
    state: HazardState,
    config: OracleCostConfig | None = None,
) -> np.ndarray:
    """Project ground-truth heat/smoke onto graph nodes as a positive proxy.

    Nodes in the extraction grid's exterior margin are ambient and smoke-free.
    The formula is a dimensionless pre-ISO routing proxy, not a tenability or
    survival claim.
    """

    cfg = config or OracleCostConfig()
    positions = graph.world_positions()
    temperature = state.sample_many(
        domain,
        positions,
        "temperature",
        outside_value=cfg.ambient_temperature_k,
    )
    smoke = state.sample_many(domain, positions, "smoke", outside_value=0.0)
    heat_burden = np.maximum(temperature - cfg.heat_onset_k, 0.0) / cfg.heat_scale_k
    smoke_burden = smoke / cfg.smoke_scale_kg_m3
    cost = (
        cfg.base_step_cost
        + cfg.heat_weight * heat_burden
        + cfg.smoke_weight * smoke_burden
    )
    return np.asarray(cost, dtype=np.float64)

@dataclass(frozen=True, slots=True)
class OptimismDiagnostic:
    """Trajectory-level Jensen gap for the passive LMDP dynamics."""

    expected_cost: np.ndarray
    value: np.ndarray
    gap: np.ndarray
    finite_mask: np.ndarray

    def __post_init__(self) -> None:
        expected = _readonly(self.expected_cost, np.dtype(np.float64))
        value = _readonly(self.value, np.dtype(np.float64))
        gap = _readonly(self.gap, np.dtype(np.float64))
        finite = _readonly(self.finite_mask, np.dtype(np.bool_))
        if not (
            expected.shape == value.shape == gap.shape == finite.shape
            and expected.ndim == 1
        ):
            raise ValueError("optimism diagnostic arrays must be matching vectors")
        if np.any(gap[finite] < -1.0e-9):
            raise ValueError("LMDP optimism gap violates Jensen's inequality")
        object.__setattr__(self, "expected_cost", expected)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gap", gap)
        object.__setattr__(self, "finite_mask", finite)

    @property
    def mean_gap(self) -> float:
        if not np.any(self.finite_mask):
            return math.inf
        return float(np.mean(self.gap[self.finite_mask]))

    @property
    def maximum_gap(self) -> float:
        if not np.any(self.finite_mask):
            return math.inf
        return float(np.max(self.gap[self.finite_mask]))


def trajectory_optimism(
    model: LMDPModel,
    solution: LMDPSolution,
) -> OptimismDiagnostic:
    """Measure E_p[D] - V under the passive trajectory distribution."""

    if solution.desirability.shape != (model.n_states,):
        raise ValueError("solution does not belong to this model")
    reachable = _states_reaching_exit(model)
    transient = np.flatnonzero(
        reachable & ~(model.exit_mask | model.death_mask)
    )
    expected = np.full(model.n_states, np.inf, dtype=np.float64)
    expected[model.exit_mask | model.death_mask] = 0.0
    if transient.size:
        local = {int(state): index for index, state in enumerate(transient)}
        matrix = np.eye(transient.size, dtype=np.float64)
        for row_index, state in enumerate(transient):
            row = model.row(int(state))
            for target, probability in zip(
                model.col_idx[row],
                model.passive_prob[row],
                strict=True,
            ):
                column = local.get(int(target))
                if column is not None:
                    matrix[row_index, column] -= float(probability)
        expected[transient] = np.linalg.solve(
            matrix,
            model.state_cost[transient],
        )
    finite = np.isfinite(expected) & np.isfinite(solution.value)
    gap = np.full(model.n_states, np.inf, dtype=np.float64)
    gap[finite] = expected[finite] - solution.value[finite]
    gap[finite & (gap < 0.0) & (gap > -1.0e-9)] = 0.0
    return OptimismDiagnostic(expected, solution.value, gap, finite)
