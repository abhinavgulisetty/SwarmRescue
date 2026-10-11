"""Independent, integer-only checks of static deadline-routing evidence.

No production solver, NumPy, MILP or benchmark oracle is imported here. Bellman
tables are proof witnesses for finite instances, not a formal verification of C.
"""

from __future__ import annotations

import gzip
import json
from itertools import pairwise
from pathlib import Path


class CertificateError(ValueError):
    """An input, recurrence or claimed route does not satisfy its contract."""


def require(condition, message):
    if not condition:
        raise CertificateError(message)


def validate_model(model):
    adjacent = model["adjacent"]
    n = len(adjacent)
    require(n > 0, "empty model")
    require(type(model["deadline"]) is int and model["deadline"] >= 0, "deadline")
    require(type(model["start"]) is int and 0 <= model["start"] < n, "start")
    require(bool(model["goals"]), "missing goals")
    require(all(type(v) is int and 0 <= v < n for v in model["goals"]), "goals")
    for row in adjacent:
        targets = set()
        for v, duration, cost in row:
            require(type(v) is int and 0 <= v < n, "arc target")
            require(type(duration) is int and duration > 0, "arc duration")
            require(type(cost) is int and cost > 0, "arc cost")
            require(v not in targets, "node-only witnesses require unique arcs")
            targets.add(v)


def bellman_table(model):
    """Construct remaining-budget optima using arbitrary-precision Python ints."""
    validate_model(model)
    adjacent, deadline = model["adjacent"], model["deadline"]
    goals = set(model["goals"])
    values = []
    for budget in range(deadline + 1):
        row = []
        for u, neighbours in enumerate(adjacent):
            if u in goals:
                row.append(0)
                continue
            choices = []
            for v, duration, cost in neighbours:
                if duration <= budget:
                    child = values[budget - duration][v]
                    if child is not None:
                        choices.append(cost + child)
            row.append(min(choices, default=None))
        values.append(row)
    return values


def verify_table(model, values):
    """Check every Bellman equality; induction on budget establishes optimality."""
    validate_model(model)
    adjacent, deadline = model["adjacent"], model["deadline"]
    goals = set(model["goals"])
    require(len(values) == deadline + 1, "table budget count")
    for budget, row in enumerate(values):
        require(len(row) == len(adjacent), "table node count")
        for u, actual in enumerate(row):
            require(
                actual is None or (type(actual) is int and actual >= 0), "table value"
            )
            if u in goals:
                expected = 0
            else:
                best = None
                for v, duration, cost in adjacent[u]:
                    if duration <= budget:
                        child = values[budget - duration][v]
                        if child is not None:
                            proposal = cost + child
                            if best is None or proposal < best:
                                best = proposal
                expected = best
            require(actual == expected, f"Bellman mismatch at ({u}, {budget})")
    return values[deadline][model["start"]]


def verify_result(model, result, optimum, *, arena_bytes=None):
    """Validate declared feasibility, exactness, lower bounds and arena accounting."""
    if arena_bytes is not None and "arena_high_water_bytes" in result:
        peak = result["arena_high_water_bytes"]
        require(type(peak) is int and 0 <= peak <= arena_bytes, "arena overrun")
    if result["feasible"]:
        route = result["route"]
        require(bool(route) and route[0] == model["start"], "route start")
        require(route[-1] in model["goals"], "route goal")
        require(len(set(route)) == len(route), "cyclic witness")
        cost = elapsed = 0
        for u, v in pairwise(route):
            require(type(u) is int and 0 <= u < len(model["adjacent"]), "route node")
            matches = [arc for arc in model["adjacent"][u] if arc[0] == v]
            require(len(matches) == 1, "missing or ambiguous route arc")
            _, duration, weight = matches[0]
            cost += weight
            elapsed += duration
        require(elapsed <= model["deadline"], "route exceeds deadline")
        require(cost == result["cost_q16"], "reported route cost")
        if "travel_steps" in result:
            require(elapsed == result["travel_steps"], "reported traversal time")
        require(
            optimum is not None and cost >= optimum, "route beats certified optimum"
        )
        lower = result.get("lower_bound_q16")
        if lower is not None:
            require(type(lower) is int and 0 <= lower <= optimum, "invalid lower bound")
        if result["optimal"]:
            require(cost == optimum, "false optimality claim")
    elif result["optimal"]:
        require(optimum is None, "false infeasibility proof")
    # feasible=False, optimal=False means no solution reported, not infeasibility.


def read_json(path):
    path = Path(path)
    content = path.read_bytes()
    if path.suffix == ".gz":
        content = gzip.decompress(content)
    return json.loads(content)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, separators=(",", ":")) + "\n").encode()
    if path.suffix == ".gz":
        content = gzip.compress(content, mtime=0)
    path.write_bytes(content)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificates", type=Path)
    args = parser.parse_args()
    evidence = read_json(args.certificates)
    states = 0
    results = 0
    for certificate in evidence["maps"]:
        model = certificate["model"]
        optimum = verify_table(model, certificate["values"])
        require(optimum == certificate["optimum_q16"], "certificate optimum")
        states += len(certificate["values"]) * len(model["adjacent"])
        for record in certificate["records"]:
            verify_result(
                model,
                record["result"],
                optimum,
                arena_bytes=evidence["numeric_arena_bytes"],
            )
            results += 1
    print(
        json.dumps(
            {
                "verified_maps": len(evidence["maps"]),
                "verified_bellman_states": states,
                "verified_results": results,
            }
        )
    )


if __name__ == "__main__":
    main()
