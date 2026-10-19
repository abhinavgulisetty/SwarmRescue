"""Synthetic fixed-source-budget rescue certificates; no production integration.

Model: h[k] = A**k B theta + base[k], A,B >= 0, theta >= 0,
sum(theta) <= M. One theta persists throughout a complete rescue schedule.
Doses are rectangular-rule occupancies at the beginning of each tick. These
are mathematical synthetic fields, not calibrated smoke or medical survival.
The standard support-function calculation below is not a novelty claim.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RescueSchedule:
    occupancy: np.ndarray  # agent order: robot, survivor; then tick, node
    pickup_tick: int
    completion_tick: int


def rescue_schedule(
    approach, escort, n_nodes, *, delay=0, approach_ticks=1, escort_ticks=2
):
    """Hold each edge's origin for its travel ticks; omit exposure after exit.

    The survivor waits at the victim through predeparture and approach, then
    follows the robot along the slower escort. Paths refer to a complete graph
    in this demonstrator; no collision or physical traversability is claimed.
    """
    approach, escort = tuple(approach), tuple(escort)
    integers = (n_nodes, delay, approach_ticks, escort_ticks, *approach, *escort)
    if any(not isinstance(x, (int, np.integer)) for x in integers):
        raise ValueError("schedule arguments must be integers")
    if n_nodes < 1 or delay < 0 or min(approach_ticks, escort_ticks) < 1:
        raise ValueError("invalid schedule durations or node count")
    if not approach or not escort or approach[-1] != escort[0]:
        raise ValueError("approach must finish at the escort's victim node")
    if min((*approach, *escort)) < 0 or max((*approach, *escort)) >= n_nodes:
        raise ValueError("schedule node outside the model")
    robot = [approach[0]] * delay
    robot += [node for node in approach[:-1] for _ in range(approach_ticks)]
    pickup = len(robot)
    survivor = [approach[-1]] * pickup
    escorted = [node for node in escort[:-1] for _ in range(escort_ticks)]
    robot += escorted
    survivor += escorted
    if not robot:
        raise ValueError("schedule must contain at least one exposure tick")
    occupancy = np.zeros((2, len(robot), n_nodes))
    for agent, nodes in enumerate((robot, survivor)):
        occupancy[agent, np.arange(len(nodes)), nodes] = 1.0
    return RescueSchedule(occupancy, pickup, len(robot))


def _inputs(transition, injection, occupancy, budget, dt, base):
    transition, injection, occupancy = (
        np.asarray(value, dtype=float) for value in (transition, injection, occupancy)
    )
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition must be square")
    n_nodes = len(transition)
    if injection.ndim != 2 or injection.shape[0] != n_nodes or not injection.shape[1]:
        raise ValueError("injection must have one row per node and nonempty sources")
    if occupancy.ndim != 3 or occupancy.shape[2] != n_nodes:
        raise ValueError("occupancy must have shape (agents, ticks, nodes)")
    if not occupancy.shape[0] or not occupancy.shape[1]:
        raise ValueError("occupancy must contain agents and exposure ticks")
    base = np.zeros(occupancy.shape[1:]) if base is None else np.asarray(base)
    if base.shape == (n_nodes,):
        base = np.broadcast_to(base, occupancy.shape[1:])
    if base.shape != occupancy.shape[1:]:
        raise ValueError("base must have shape (ticks, nodes) or (nodes,)")
    for array in (transition, injection, occupancy, base):
        if not np.all(np.isfinite(array)) or np.any(array < 0):
            raise ValueError("model, base, and occupancy must be finite nonnegative")
    if np.any(occupancy.sum(axis=2) > 1 + 1e-12):
        raise ValueError("each agent can occupy at most one unit of space per tick")
    if not np.isfinite(budget) or budget < 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("source budget must be nonnegative and dt positive")
    return transition, injection, occupancy, base


@dataclass(frozen=True)
class DoseCertificate:
    base_dose: np.ndarray
    coefficients: np.ndarray
    worst_case: np.ndarray
    pointwise_reset: np.ndarray
    current_snapshot: np.ndarray


def dose_certificate(transition, injection, occupancy, budget, *, dt=1.0, base=None):
    """Accumulate q and apply the simplex support function b + M max(q).

    Pointwise reset permits a different source vertex at every tick. Snapshot
    freezes A**k B at k=0, while retaining the full known deterministic base.
    Both comparison methods use precisely the same complete occupancies.
    """
    transition, injection, occupancy, base = _inputs(
        transition, injection, occupancy, budget, dt, base
    )
    coefficients = np.zeros((len(occupancy), injection.shape[1]))
    pointwise = np.zeros(len(occupancy))
    field = injection.copy()
    for tick in range(occupancy.shape[1]):
        per_tick = occupancy[:, tick] @ field * dt
        coefficients += per_tick
        pointwise += per_tick.max(axis=1)
        field = transition @ field
    base_dose = np.einsum("atn,tn->a", occupancy, base) * dt
    snapshot = occupancy.sum(axis=1) @ injection * dt
    return DoseCertificate(
        base_dose,
        coefficients,
        base_dose + budget * coefficients.max(axis=1),
        base_dose + budget * pointwise,
        base_dose + budget * snapshot.max(axis=1),
    )


def forward_vertex_oracle(
    transition, injection, occupancy, budget, *, dt=1.0, base=None
):
    """Independently simulate hazard vectors at zero and every simplex vertex.

    This deliberately never constructs A**k B or calls dose_certificate.
    """
    transition, injection, occupancy, base = _inputs(
        transition, injection, occupancy, budget, dt, base
    )
    vertices = np.vstack(
        (np.zeros(injection.shape[1]), budget * np.eye(injection.shape[1]))
    )
    doses = []
    for theta in vertices:
        hazard = injection @ theta
        dose = np.zeros(len(occupancy))
        for tick in range(occupancy.shape[1]):
            for agent in range(len(occupancy)):
                dose[agent] += (
                    float(occupancy[agent, tick] @ (hazard + base[tick])) * dt
                )
            hazard = transition @ hazard
        doses.append(dose)
    return np.max(doses, axis=0)


def quantized_upper(base_dose, coefficients, budget, delta, *, base_delta=0.0):
    """Upward rounding gives safety and additive error <= M*delta + base_delta."""
    base_dose, coefficients = np.asarray(base_dose), np.asarray(coefficients)
    if not np.isfinite(delta) or delta <= 0 or not np.isfinite(base_delta):
        raise ValueError("coefficient quantum must be positive and finite")
    if base_delta < 0 or not np.isfinite(budget) or budget < 0:
        raise ValueError("base quantum and source budget must be nonnegative")
    if coefficients.ndim != 2 or base_dose.shape != (len(coefficients),):
        raise ValueError("base and coefficient agent dimensions must match")
    for value in (base_dose, coefficients):
        if not np.all(np.isfinite(value)) or np.any(value < 0):
            raise ValueError("dose coefficients must be finite nonnegative")
    rounded = np.ceil(coefficients / delta) * delta
    rounded_base = (
        np.ceil(base_dose / base_delta) * base_delta if base_delta else base_dose
    )
    return rounded_base + budget * rounded.max(axis=1)


def distinguish_coefficients(left, right, bound):
    """A fixed-threshold continuation distinguishes any different integer q.

    Domain is unrestricted coefficient vectors {0,...,bound}**m and arbitrary
    nonnegative continuations, not vectors realizable by a given physical model.
    Return r_i=bound-min(q_i,q'_i), all other r=0; threshold is bound.
    """
    left, right = tuple(left), tuple(right)
    if not isinstance(bound, int) or bound < 0 or not left or len(left) != len(right):
        raise ValueError("invalid integer coefficient domain")
    if any(
        not isinstance(x, (int, np.integer)) or not 0 <= x <= bound
        for x in (*left, *right)
    ):
        raise ValueError("coefficients must lie in the integer domain")
    coordinate = next(
        (
            i
            for i, pair in enumerate(zip(left, right, strict=True))
            if pair[0] != pair[1]
        ),
        None,
    )
    if coordinate is None:
        raise ValueError("coefficient vectors must differ")
    continuation = np.zeros(len(left), dtype=int)
    continuation[coordinate] = bound - min(left[coordinate], right[coordinate])
    return continuation


def packed_summary(coefficients, bound):
    """Exact radix-(bound+1) packing attains ceil(log2((bound+1)**m)) bits."""
    values = tuple(coefficients)
    if not values or not isinstance(bound, int) or bound < 0:
        raise ValueError("invalid integer coefficient domain")
    if any(not isinstance(x, (int, np.integer)) or not 0 <= x <= bound for x in values):
        raise ValueError("coefficients must lie in the integer domain")
    packed = sum(int(value) * (bound + 1) ** i for i, value in enumerate(values))
    return packed, ((bound + 1) ** len(values) - 1).bit_length()


def exhaustive_checks():
    """Exhaustive small-domain algebra and continuation checks, not smoke data."""
    support_errors, reset_violations = [], 0
    for flat in itertools.product(range(3), repeat=6):
        ticks = np.array(flat).reshape(3, 2)
        q = ticks.sum(axis=0)
        support = 2 * q.max()
        vertex = max(
            sum(float(row @ theta) for row in ticks)
            for theta in (np.zeros(2), np.array([2, 0]), np.array([0, 2]))
        )
        support_errors.append(abs(support - vertex))
        reset_violations += int(2 * ticks.max(axis=1).sum() < support)
    domain = list(itertools.product(range(3), repeat=3))
    distinction_violations = 0
    for left, right in itertools.combinations(domain, 2):
        continuation = distinguish_coefficients(left, right, 2)
        distinction_violations += int(
            (np.max(left + continuation) <= 2) == (np.max(right + continuation) <= 2)
        )
    summaries = [packed_summary(q, 2)[0] for q in domain]
    return {
        "coefficient_tick_cases": len(support_errors),
        "support_max_abs_error": max(support_errors),
        "pointwise_dominance_violations": reset_violations,
        "integer_domain": {
            "dimension": 3,
            "bound": 2,
            "distinct_vectors": 27,
            "pair_checks": 351,
            "fixed_length_bits": 5,
            "distinction_violations": distinction_violations,
            "distinct_packed_summaries": len(set(summaries)),
        },
    }


def sharp_witnesses():
    q, other, future = np.array([1, 0]), np.array([0, 1]), np.array([1, 0])
    waiting = rescue_schedule((0, 1), (1, 0), 2, delay=2, escort_ticks=3)
    wait_dose = dose_certificate(
        np.eye(2), np.array([[0.0], [1.0]]), waiting.occupancy, 1.0
    )
    growth = dose_certificate(
        np.array([[2.0]]), np.array([[1.0]]), np.ones((1, 3, 1)), 1.0
    )
    plume_transition, plume_injection = np.zeros((6, 6)), np.zeros((6, 1))
    plume_transition[1, 2] = plume_transition[2, 3] = plume_transition[3, 4] = 1
    plume_injection[4, 0] = 1
    delayed_doses = []
    for delay in range(3):
        schedule = rescue_schedule((5, 0), (0, 1, 5), 6, delay=delay, escort_ticks=1)
        delayed_doses.append(
            float(
                dose_certificate(
                    plume_transition, plume_injection, schedule.occupancy, 1
                ).worst_case[1]
            )
        )
    return {
        "scalar_noncomposability": {
            "q": q.tolist(),
            "q_prime": other.tolist(),
            "continuation": future.tolist(),
            "prefix_maxima": [1, 1],
            "complete_maxima": [int(max(q + future)), int(max(other + future))],
        },
        "pointwise_gap": {
            "tick_coefficients": [[1, 0], [0, 1]],
            "fixed_source_worst_case": 1,
            "pointwise_reset": 2,
            "threshold": 1,
            "fixed_source_accepts": True,
            "pointwise_accepts": False,
        },
        "snapshot_unsafe": {
            "transition": [[2]],
            "horizon": 3,
            "worst_case": float(growth.worst_case[0]),
            "snapshot": float(growth.current_snapshot[0]),
            "threshold": 4,
        },
        "waiting_exposure": {
            "pickup_tick": waiting.pickup_tick,
            "completion_tick": waiting.completion_tick,
            "robot_dose": float(wait_dose.worst_case[0]),
            "survivor_dose": float(wait_dose.worst_case[1]),
            "survivor_dose_if_waiting_omitted": 3,
        },
        "nonmonotone_delay": {
            "transition_nonzero_edges": [[1, 2], [2, 3], [3, 4]],
            "source_node": 4,
            "approach": [5, 0],
            "escort": [0, 1, 5],
            "approach_ticks": 1,
            "escort_ticks": 1,
            "delays": [0, 1, 2],
            "survivor_doses": delayed_doses,
            "threshold": 0.5,
            "feasible": [dose <= 0.5 for dose in delayed_doses],
        },
    }


def random_study(seed=20261016, systems=200, schedules_per_system=8):
    """Same-mission alternatives in an oracle-derived synthetic acceptance sweep.

    Thresholds are derived from these very schedules, not a heldout protocol.
    """
    if systems < 1 or schedules_per_system < 2:
        raise ValueError("study needs positive systems and at least two schedules")
    rng = np.random.default_rng(seed)
    names = ("fixed_source", "pointwise_reset", "current_snapshot", "quantized_upper")
    counts = {
        name: dict(
            accepted=0,
            unsafe_accepts=0,
            feasible_rejections=0,
            selected_systems=0,
            unsafe_selected=0,
        )
        for name in names
    }
    max_error = max_quantization_error = max_pointwise_ratio = 0.0
    dominance_violations = quantization_violations = 0
    feasible_count = 0
    for system in range(systems):
        n_nodes, n_sources = 6, 3
        transition = rng.uniform(0.0, 0.025, (n_nodes, n_nodes))
        if system % 2:
            transition += np.eye(n_nodes) * rng.uniform(1.0, 1.04)
        else:
            transition += np.eye(n_nodes) * 0.5
            transition *= rng.uniform(0.9, 1.05, n_nodes) / transition.sum(axis=0)
        injection = rng.uniform(0.0, 1.0, (n_nodes, n_sources))
        injection[rng.random(injection.shape) < 0.45] = 0
        budget, dt, delta, base_delta = float(rng.uniform(0.5, 2.0)), 0.25, 0.05, 0.01
        base = rng.uniform(0.0, 0.08, n_nodes)
        start, victim, exit_node = rng.choice(n_nodes, 3, replace=False)
        records = []
        for _ in range(schedules_per_system):
            middle = tuple(map(int, rng.integers(0, n_nodes, int(rng.integers(0, 3)))))
            approach, escort = (
                (int(start), *middle, int(victim)),
                (int(victim), int(exit_node)),
            )
            schedule = rescue_schedule(
                approach,
                escort,
                n_nodes,
                delay=int(rng.integers(0, 5)),
                escort_ticks=int(rng.integers(2, 5)),
            )
            cert = dose_certificate(
                transition, injection, schedule.occupancy, budget, dt=dt, base=base
            )
            oracle = forward_vertex_oracle(
                transition, injection, schedule.occupancy, budget, dt=dt, base=base
            )
            quantized = quantized_upper(
                cert.base_dose, cert.coefficients, budget, delta, base_delta=base_delta
            )
            max_error = max(max_error, float(np.max(np.abs(cert.worst_case - oracle))))
            error = quantized - cert.worst_case
            max_quantization_error = max(max_quantization_error, float(error.max()))
            quantization_violations += int(
                np.any(error < -1e-10)
                or np.any(error > budget * delta + base_delta + 1e-10)
            )
            dominance_violations += int(np.any(cert.pointwise_reset < oracle - 1e-10))
            max_pointwise_ratio = max(
                max_pointwise_ratio,
                float(np.max(cert.pointwise_reset / np.maximum(oracle, 1e-15))),
            )
            records.append(
                (
                    schedule.completion_tick,
                    oracle,
                    (
                        cert.worst_case,
                        cert.pointwise_reset,
                        cert.current_snapshot,
                        quantized,
                    ),
                )
            )
        limits = np.quantile(np.array([row[1] for row in records]), 0.6, axis=0)
        decisions = {name: [] for name in names}
        for completion, oracle, methods in records:
            feasible = bool(np.all(oracle <= limits + 1e-10))
            feasible_count += int(feasible)
            for name, values in zip(names, methods, strict=True):
                accepts = bool(np.all(values <= limits + 1e-10))
                counts[name]["accepted"] += int(accepts)
                counts[name]["unsafe_accepts"] += int(accepts and not feasible)
                counts[name]["feasible_rejections"] += int(not accepts and feasible)
                if accepts:
                    decisions[name].append((completion, not feasible))
        for name, options in decisions.items():
            if options:
                counts[name]["selected_systems"] += 1
                chosen = min(options, key=lambda option: option[0])
                counts[name]["unsafe_selected"] += int(chosen[1])
    return {
        "systems": systems,
        "schedules": systems * schedules_per_system,
        "agent_doses": 2 * systems * schedules_per_system,
        "forward_vertex_trajectories": 4 * systems * schedules_per_system,
        "oracle_feasible_schedules": feasible_count,
        "max_abs_error": max_error,
        "pointwise_dominance_violations": dominance_violations,
        "max_pointwise_to_exact_ratio": max_pointwise_ratio,
        "quantization_bound_violations": quantization_violations,
        "max_quantization_upper_error": max_quantization_error,
        "rejection_tradeoff": counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--systems", type=int, default=200)
    parser.add_argument("--schedules-per-system", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=Path("docs/research/source-budget-results.json")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = (Path(__file__).resolve(), root / "tests/test_source_budget_rescue.py")
    result = {
        "scope": "Synthetic positive linear fields; no calibrated smoke, clinical, "
        "physical rank lower-bound, production-integration, or novelty claim.",
        "seed": args.seed,
        "protocol": {
            "version": 1,
            "model": "h[k]=A^k B theta+base[k]",
            "source_set": "theta>=0, sum(theta)<=M; one fixed theta per mission",
            "graph": "complete, synthetic",
            "agents": ["robot", "survivor"],
            "random_regimes": ["transport/decay", "monotone growth"],
            "dt": 0.25,
            "coefficient_quantum": 0.05,
            "base_quantum": 0.01,
            "limits": "per-agent 0.6 quantile across paired schedules in each system",
            "design": "oracle-derived synthetic acceptance sweep; no heldout "
            "improvement claim; start/victim/exit fixed per system",
            "selection": "minimum completion tick among accepted schedules",
            "snapshot": "freezes source field at k=0; keeps deterministic base",
            "information_bound_domain": "all integer q in {0,...,B}^m and "
            "unrestricted nonnegative additive continuations; fixed threshold B",
        },
        "exhaustive": exhaustive_checks(),
        "witnesses": sharp_witnesses(),
        "random": random_study(args.seed, args.systems, args.schedules_per_system),
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
            if path.exists()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **result["random"]}, sort_keys=True))


if __name__ == "__main__":
    main()
