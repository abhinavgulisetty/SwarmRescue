"""Independent time-expanded graph validation of complete rescue certificates.

NetworkX supplies the generic DAG longest-path algorithm.  This module does not
import or reproduce the production certificate's optimized recurrence.  Its
synthetic node/tick exposure model is not a smoke or human-survival calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class GraphCertificate:
    robot_bound: float | int
    survivor_bound: float | int
    robot_scenario: int
    survivor_scenario: int
    robot_durations: tuple[int, ...]
    survivor_durations: tuple[int, ...]
    latest_completion: int
    graph_nodes: int
    graph_edges: int


def _durations(value, count, name):
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        result = (int(value),) * count
    else:
        result = tuple(value)
    if len(result) != count or any(
        isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1
        for v in result
    ):
        raise ValueError(f"{name} must contain one positive integer per edge")
    return tuple(int(v) for v in result)


def _normalize(scenarios, approach, escort, approach_ticks, low, high, delay, dt):
    field = np.asarray(scenarios, dtype=float)
    if field.ndim != 3 or field.shape[0] < 1 or field.shape[2] < 1:
        raise ValueError(
            "scenarios must have shape (scenario,tick,node) with scenarios and nodes"
        )
    if not np.all(np.isfinite(field)) or np.any(field < 0):
        raise ValueError("exposure must be finite and nonnegative")
    approach, escort = tuple(approach), tuple(escort)
    if not approach or not escort or approach[-1] != escort[0]:
        raise ValueError("approach must end at the escort's first node")
    if any(
        not isinstance(v, (int, np.integer)) or v < 0 or v >= field.shape[2]
        for v in (*approach, *escort)
    ):
        raise ValueError("route node outside the field")
    if not isinstance(delay, (int, np.integer)) or isinstance(delay, bool) or delay < 0:
        raise ValueError("departure delay must be a nonnegative integer")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    walk = _durations(approach_ticks, len(approach) - 1, "approach_ticks")
    low = _durations(low, len(escort) - 1, "escort_min_ticks")
    high = (
        low if high is None else _durations(high, len(escort) - 1, "escort_max_ticks")
    )
    if any(a > b for a, b in zip(low, high, strict=True)):
        raise ValueError("minimum escort dwell exceeds maximum")
    pickup = int(delay) + sum(walk)
    latest = pickup + sum(high)
    if latest > field.shape[1]:
        raise ValueError("hazard horizon does not contain every admissible schedule")
    return field, approach, escort, walk, low, high, int(delay), pickup, latest


def _approach_nodes(approach, walk, delay):
    return [approach[0]] * delay + [
        node
        for node, ticks in zip(approach[:-1], walk, strict=True)
        for _ in range(ticks)
    ]


def graph_oracle(
    scenarios,
    approach,
    escort,
    *,
    approach_ticks=1,
    escort_min_ticks=1,
    escort_max_ticks=None,
    departure_delay=0,
    dt=1.0,
):
    """Maximize over every lag schedule using explicit generic NetworkX graphs.

    An edge holds the origin node for its duration, including travel.  Each
    scenario has an entirely separate graph.  The +1 in the optimization weight
    forces longest paths to include source and sink; every complete path has
    the same number of edges, so it cannot change the exposure maximizer.
    Returned bounds are direct sums of the unshifted edge exposure attributes.
    For integer input rates and dt=1, all graph weights and accumulation use
    Python integers, independently of the kernel's float64 implementation.
    """
    raw_field = np.asarray(scenarios)
    integer_units = np.issubdtype(raw_field.dtype, np.integer) and dt == 1
    f, approach, escort, walk, low, high, delay, pickup, latest = _normalize(
        scenarios,
        approach,
        escort,
        approach_ticks,
        escort_min_ticks,
        escort_max_ticks,
        departure_delay,
        dt,
    )
    robot_nodes = _approach_nodes(approach, walk, delay)
    records = []
    total_nodes = total_edges = 0
    for source in range(len(f)):
        graph = nx.DiGraph()
        start, finish = ("source",), ("refuge",)
        zero = 0 if integer_units else 0.0
        graph.add_edge(start, (0, pickup), weight=1, exposure=zero, duration=None)
        reachable = {pickup}
        for stage, node in enumerate(escort[:-1]):
            following = set()
            for tick in sorted(reachable):
                for dwell in range(low[stage], high[stage] + 1):
                    end = tick + dwell
                    exposure = (
                        sum(int(raw_field[source, u, node]) for u in range(tick, end))
                        if integer_units
                        else float(
                            sum(f[source, u, node] * dt for u in range(tick, end))
                        )
                    )
                    graph.add_edge(
                        (stage, tick),
                        (stage + 1, end),
                        weight=exposure + 1,
                        exposure=exposure,
                        duration=dwell,
                    )
                    following.add(end)
            reachable = following
        for tick in reachable:
            graph.add_edge(
                (len(low), tick), finish, weight=1, exposure=zero, duration=None
            )
        path = nx.dag_longest_path(graph, weight="weight")
        if path[0] != start or path[-1] != finish:
            raise AssertionError(
                "generic longest-path witness omitted source or refuge"
            )
        arcs = [graph.edges[a, b] for a, b in itertools.pairwise(path)]
        escort_dose = sum(edge["exposure"] for edge in arcs)
        dwell = tuple(edge["duration"] for edge in arcs if edge["duration"] is not None)
        if integer_units:
            robot_base = sum(
                int(raw_field[source, tick, node])
                for tick, node in enumerate(robot_nodes)
            )
            survivor_base = sum(
                int(raw_field[source, tick, approach[-1]]) for tick in range(pickup)
            )
        else:
            robot_base = float(
                sum(f[source, tick, node] * dt for tick, node in enumerate(robot_nodes))
            )
            survivor_base = float(
                sum(f[source, tick, approach[-1]] * dt for tick in range(pickup))
            )
        records.append((robot_base + escort_dose, survivor_base + escort_dose, dwell))
        total_nodes += graph.number_of_nodes()
        total_edges += graph.number_of_edges()
    robot_source = max(range(len(records)), key=lambda s: records[s][0])
    survivor_source = max(range(len(records)), key=lambda s: records[s][1])
    return GraphCertificate(
        records[robot_source][0],
        records[survivor_source][1],
        robot_source,
        survivor_source,
        records[robot_source][2],
        records[survivor_source][2],
        latest,
        total_nodes,
        total_edges,
    )


def replay_mission(
    scenarios,
    approach,
    escort,
    durations,
    *,
    approach_ticks=1,
    departure_delay=0,
    dt=1.0,
):
    """Directly replay start, waiting, contact, escort and final refuge arrival."""
    f, approach, escort, walk, _, _, delay, pickup, latest = _normalize(
        scenarios,
        approach,
        escort,
        approach_ticks,
        durations,
        durations,
        departure_delay,
        dt,
    )
    robot = _approach_nodes(approach, walk, delay)
    survivor = [approach[-1]] * pickup
    guided = [
        node
        for node, dwell in zip(escort[:-1], durations, strict=True)
        for _ in range(dwell)
    ]
    robot += guided
    survivor += guided
    doses = np.zeros((len(f), 2))
    for source in range(len(f)):
        for tick in range(latest):
            doses[source, 0] += f[source, tick, robot[tick]] * dt
            doses[source, 1] += f[source, tick, survivor[tick]] * dt
    return {
        "doses": doses,
        "pickup_tick": pickup,
        "completion_tick": latest,
        "robot_tick_nodes": tuple(robot),
        "survivor_tick_nodes": tuple(survivor),
        "terminal_node": escort[-1],
    }


def exhaustive_replay(
    scenarios, approach, escort, *, escort_min_ticks=1, escort_max_ticks=None, **kwargs
):
    """Enumerate lag tuples independently; intended for small verification cases."""
    low = _durations(escort_min_ticks, len(escort) - 1, "escort_min_ticks")
    high = (
        low
        if escort_max_ticks is None
        else _durations(escort_max_ticks, len(low), "escort_max_ticks")
    )
    # Validate the full product before enumeration; never silently drop late paths.
    _normalize(
        scenarios,
        approach,
        escort,
        kwargs.get("approach_ticks", 1),
        low,
        high,
        kwargs.get("departure_delay", 0),
        kwargs.get("dt", 1.0),
    )
    bound = np.zeros(2)
    count = 0
    for durations in itertools.product(
        *(range(a, b + 1) for a, b in zip(low, high, strict=True))
    ):
        replay = replay_mission(scenarios, approach, escort, durations, **kwargs)
        bound = np.maximum(bound, replay["doses"].max(axis=0))
        count += len(scenarios)
    return bound, count


def frozen_cases(seed=20261016, missions=96):
    """Fixed connected grid missions; parameters are not tuned after outcomes."""
    rng = np.random.default_rng(seed)
    for mission in range(missions):
        graph = nx.grid_2d_graph(4, 4)
        graph = nx.relabel_nodes(graph, {(x, y): 4 * x + y for x, y in graph})
        victim = int(rng.integers(8, 16))
        refuge = 3
        removable = [v for v in graph if v not in (0, refuge, victim)]
        blocked = int(rng.choice(removable))
        graph.remove_node(blocked)
        assert nx.is_connected(graph)
        approach = tuple(nx.shortest_path(graph, 0, victim))
        escorts = list(
            itertools.islice(nx.shortest_simple_paths(graph, victim, refuge), 3)
        )
        field = np.zeros((3, 48, 16))
        ticks = np.arange(48)
        for source in range(3):
            center = int(rng.integers(0, 16))
            distance = np.array(
                [abs(v // 4 - center // 4) + abs(v % 4 - center % 4) for v in range(16)]
            )
            spatial = np.exp(-distance / 1.8)
            peak = int(rng.integers(3, 32))
            kind = (mission + source) % 3
            if kind == 0:
                temporal = np.exp(-(((ticks - peak) / 3.0) ** 2))
            elif kind == 1:
                temporal = np.exp(-ticks / float(rng.uniform(5, 18)))
            else:
                temporal = np.minimum(ticks / float(rng.uniform(12, 32)), 1.0)
            field[source] = (
                0.03 + float(rng.uniform(0.6, 2.4)) * temporal[:, None] * spatial
            )
        field[:, :, refuge] = 0.0  # Declared absorbing refuge, not inferred safety.
        budgets = ((3.0, 4.0), (5.0, 6.0), (7.0, 9.0))[mission % 3]
        for candidate, escort in enumerate(escorts):
            low = tuple(1 + (i + mission) % 2 for i in range(len(escort) - 1))
            high = tuple(a + 1 for a in low)
            yield {
                "mission": mission,
                "candidate": candidate,
                "graph": graph,
                "scenarios": field,
                "approach": approach,
                "escort": tuple(escort),
                "kwargs": {
                    "approach_ticks": 1,
                    "escort_min_ticks": low,
                    "escort_max_ticks": high,
                    "departure_delay": mission % 4,
                    "dt": 0.5,
                },
                "robot_budget": budgets[0],
                "survivor_budget": budgets[1],
            }


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_study(output_dir, *, missions=96):
    from ember.rescue_certificate import certify_integer_rescue, certify_rescue

    if not isinstance(missions, int) or isinstance(missions, bool) or missions < 1:
        raise ValueError("the study must contain at least one mission")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    protocol = {
        "research_date": "2026-10-16",
        "seed": 20261016,
        "missions": missions,
        "candidate_routes_per_mission": 3,
        "source_scenarios": 3,
        "hazard_ticks": 48,
        "grid": [4, 4],
        "removed_nodes_per_mission": 1,
        "selection": (
            "among certified candidates minimize latest completion, "
            "then survivor bound, then candidate index"
        ),
        "budgets": [[3, 4], [5, 6], [7, 9]],
        "dt": 0.5,
        "paired_integer_check": {
            "field_definition": "ceil(float_synthetic_field * 256).astype(int64)",
            "dt": 1,
            "oracle_arithmetic": "Python integer arc weights and path sums",
            "comparison": (
                "exact equality of both actor bounds; "
                "a separately defined synthetic objective"
            ),
        },
        "uncertainty": (
            "one scenario for entire rescue; independent integer escort edge "
            "durations in declared intervals"
        ),
        "scope": (
            "synthetic single-robot single-survivor complete missions; "
            "origin-held edge exposure; refuge absorbing; no physical/fleet claims"
        ),
        "dependencies": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "networkx": nx.__version__,
        },
        "source_sha256": {
            name: _sha(root / name)
            for name in (
                "eval/lag_rescue_validation.py",
                "ember/rescue_certificate.py",
            )
        },
        "test_source_sha256": (
            _sha(root / "tests/test_lag_rescue_validation.py")
            if (root / "tests/test_lag_rescue_validation.py").is_file()
            else None
        ),
        "tests_bundled_with_runtime": (
            root / "tests/test_lag_rescue_validation.py"
        ).is_file(),
        "external_method": {
            "implementation": (
                "NetworkX dag_longest_path on independently materialized "
                "stage/time graphs"
            ),
            "documentation": "https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.dag.dag_longest_path.html",
            "library_citation": (
                "Hagberg, Schult and Swart, Exploring Network Structure, "
                "Dynamics, and Function using NetworkX, SciPy 2008"
            ),
            "paper": "https://proceedings.scipy.org/articles/TCWV9851.pdf",
            "robust_routing_reference": "https://doi.org/10.1016/S0305-0548(97)00085-3",
            "adaptation": (
                "fixed-scenario routing uses each whole hazard tensor as a "
                "scenario; nominal comparator fixes all escort durations to "
                "their lower bounds, omitting robust lag"
            ),
        },
    }
    serialized = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    (output / "protocol.json").write_text(serialized)
    protocol_hash = hashlib.sha256(serialized.encode()).hexdigest()
    records, kernel_ms, graph_ms = [], [], []
    integer_exact_cases = integer_mismatches = 0
    selected = {}
    replays = []
    for case in frozen_cases(missions=missions):
        field, approach, escort, kwargs = (
            case[k] for k in ("scenarios", "approach", "escort", "kwargs")
        )
        for path in (approach, escort):
            assert all(
                case["graph"].has_edge(a, b) for a, b in itertools.pairwise(path)
            )
        budget_kwargs = {
            "robot_budget": case["robot_budget"],
            "survivor_budget": case["survivor_budget"],
        }
        started = time.perf_counter()
        certificate = certify_rescue(field, approach, escort, **kwargs, **budget_kwargs)
        kernel_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        oracle = graph_oracle(field, approach, escort, **kwargs)
        graph_ms.append((time.perf_counter() - started) * 1000)
        expected = np.array([oracle.robot_bound, oracle.survivor_bound])
        actual = np.array([certificate.robot_bound, certificate.survivor_bound])
        budgets = np.array([case["robot_budget"], case["survivor_budget"]])
        if not np.allclose(actual, expected, atol=1e-10, rtol=1e-12):
            raise AssertionError(f"kernel/oracle mismatch: {actual} != {expected}")
        truth = bool(np.all(expected <= budgets))
        if bool(certificate.feasible) != truth:
            raise AssertionError("kernel feasibility differs from exact generic graph")
        integer_field = np.ceil(field * 256).astype(np.int64)
        integer_kwargs = {**kwargs, "dt": 1}
        integer_certificate = certify_integer_rescue(
            integer_field,
            approach,
            escort,
            **integer_kwargs,
        )
        integer_oracle = graph_oracle(integer_field, approach, escort, **integer_kwargs)
        integer_equal = (
            integer_certificate.robot_bound == integer_oracle.robot_bound
            and integer_certificate.survivor_bound == integer_oracle.survivor_bound
        )
        integer_exact_cases += 1
        integer_mismatches += not integer_equal
        if not integer_equal:
            raise AssertionError(
                "integer kernel differs from arbitrary-precision NetworkX oracle"
            )
        bounds = {"lag_robust": actual.tolist()}
        for name, fixed in (
            ("fixed_timing_scenario", kwargs["escort_min_ticks"]),
            ("maximum_duration_only", kwargs["escort_max_ticks"]),
        ):
            replay = replay_mission(
                field,
                approach,
                escort,
                fixed,
                approach_ticks=kwargs["approach_ticks"],
                departure_delay=kwargs["departure_delay"],
                dt=kwargs["dt"],
            )
            bounds[name] = replay["doses"].max(axis=0).tolist()
        reset = graph_oracle(
            field.max(axis=0, keepdims=True), approach, escort, **kwargs
        )
        bounds["pointwise_scenario_reset"] = [reset.robot_bound, reset.survivor_bound]
        feasible = {
            name: bool(np.all(np.asarray(bound) <= budgets))
            for name, bound in bounds.items()
        }
        record = {
            "mission": case["mission"],
            "candidate": case["candidate"],
            "approach": list(approach),
            "escort": list(escort),
            "bounds": bounds,
            "feasible": feasible,
            "budgets": budgets.tolist(),
            "kernel_oracle_error": float(np.max(np.abs(actual - expected))),
            "integer_exact_equal": bool(integer_equal),
            "integer_unit_bounds": [
                int(integer_oracle.robot_bound),
                int(integer_oracle.survivor_bound),
            ],
            "latest_completion": oracle.latest_completion,
            "robot_witness": [oracle.robot_scenario, list(oracle.robot_durations)],
            "survivor_witness": [
                oracle.survivor_scenario,
                list(oracle.survivor_durations),
            ],
            "graph_nodes": oracle.graph_nodes,
            "graph_edges": oracle.graph_edges,
        }
        records.append(record)
        if truth:
            key = (oracle.latest_completion, oracle.survivor_bound, case["candidate"])
            previous = selected.get(case["mission"])
            if previous is None or key < previous[0]:
                selected[case["mission"]] = (key, case)
    # Every scenario and every lag tuple of selected routes, not selected traces.
    for mission, (_, case) in sorted(selected.items()):
        expected, count = exhaustive_replay(
            case["scenarios"], case["approach"], case["escort"], **case["kwargs"]
        )
        if np.any(expected > [case["robot_budget"], case["survivor_budget"]]):
            raise AssertionError(
                "a selected rescue failed an exhaustive mission replay"
            )
        replays.append(
            {
                "mission": mission,
                "candidate": case["candidate"],
                "scenario_schedule_replays": count,
                "robot_max_dose": float(expected[0]),
                "survivor_max_dose": float(expected[1]),
                "refuge": case["escort"][-1],
                "all_replays_completed": True,
            }
        )
    names = records[0]["bounds"] if records else []
    method_counts = {}
    for name in names:
        accept = [r for r in records if r["feasible"][name]]
        method_counts[name] = {
            "accepted_candidates": len(accept),
            "false_acceptances": sum(not r["feasible"]["lag_robust"] for r in accept),
            "false_rejections": sum(
                r["feasible"]["lag_robust"] and not r["feasible"][name] for r in records
            ),
        }
    result = {
        "protocol_sha256": protocol_hash,
        "candidate_cases": len(records),
        "kernel_graph_max_absolute_error": max(
            (r["kernel_oracle_error"] for r in records), default=0
        ),
        "integer_exact_cases": integer_exact_cases,
        "integer_exact_mismatches": integer_mismatches,
        "method_counts": method_counts,
        "selected_complete_missions": len(selected),
        "exhaustive_scenario_schedule_replays": sum(
            r["scenario_schedule_replays"] for r in replays
        ),
        "all_selected_replays_completed_within_budgets": True,
        "complete_call_timing_ms": {
            name: {
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
                "total": float(sum(values)),
                "calls": len(values),
            }
            for name, values in (
                ("certificate_kernel", kernel_ms),
                ("generic_networkx_graph", graph_ms),
            )
        },
        "selection_scope": (
            "best declared candidate by latest completion, then robust survivor "
            "dose; no global routing optimum or fleet scheduling claim"
        ),
        "records": records,
        "complete_mission_replays": replays,
    }
    (output / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return {
        key: value
        for key, value in result.items()
        if key not in ("records", "complete_mission_replays")
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="docs/research/lag-rescue")
    parser.add_argument("--missions", type=int, default=96)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_study(args.output_dir, missions=args.missions), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
