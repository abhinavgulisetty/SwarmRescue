"""Explain deadline-aware route selection with a deterministic, verified case."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from cascade.extract import TraversabilityGraph, VoxelGrid
from ember.routing.deadline import deadline_route


def comparison():
    heat = [2, 12, 5, 18, 4, 14, 3, 16, 6, 10]
    positions = [[stage, 0] for stage in range(11)]
    positions += [[stage + 0.5, -0.8] for stage in range(10)]
    positions += [[stage + 0.5, 0.8] for stage in range(10)]
    edges, lengths = [], []
    for stage in range(10):
        edges += [
            (stage, 11 + stage),
            (11 + stage, stage + 1),
            (stage, 21 + stage),
            (21 + stage, stage + 1),
        ]
        lengths += [1, 1, 2, 2]
    empty = np.zeros((31, 1, 1), bool)
    graph = TraversabilityGraph(
        VoxelGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), empty, empty),
        np.array([[i, 0, 0] for i in range(31)]),
        np.zeros(31),
        np.array(edges),
        np.array(lengths, float),
    )
    risk = np.array([0.0] * 11 + heat + [0.0] * 10)
    lookup = {
        tuple(sorted(edge)): length for edge, length in zip(edges, lengths, strict=True)
    }
    cases = {}
    for deadline in range(20, 41):
        rows = {}
        for name, strategy in (("legacy", "pareto"), ("hybrid", "hybrid_exact")):
            route = deadline_route(
                graph,
                0,
                (10,),
                risk,
                available_steps=deadline,
                walking_speed_m_s=1,
                dt_s=1,
                max_labels=32,
                strategy=strategy,
            )
            assert route.feasible and route.travel_steps <= deadline
            row = asdict(route)
            row["edge_steps"] = [
                lookup[tuple(sorted((a, b)))] for a, b in pairwise(route.route)
            ]
            row["estimated_exposure"] = route.estimated_objective - route.travel_steps
            rows[name] = row
        # Independent closed-form knapsack oracle: each quiet detour takes
        # two extra steps and removes risk-2 cost units. No solver reuse.
        quiet_count = (deadline - 20) // 2
        expected = (
            20
            + sum(heat)
            - sum(sorted((h - 2 for h in heat), reverse=True)[:quiet_count])
        )
        assert rows["hybrid"]["optimal"]
        assert rows["hybrid"]["estimated_objective"] == expected
        assert rows["legacy"]["estimated_objective"] >= expected
        cases[str(deadline)] = rows
    return {
        "positions": positions,
        "edges": edges,
        "heat": heat,
        "cases": cases,
        "accepted": True,
        "scope": (
            "Static estimated route cost; schematic, not a physical rescue experiment."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/deadline-demo"))
    args = parser.parse_args(argv)
    data = comparison()
    template = Path(__file__).with_name("templates") / "deadline.html"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comparison.json").write_text(json.dumps(data, indent=2) + "\n")
    (args.out / "index.html").write_text(
        template.read_text().replace("__COMPARISON_DATA__", json.dumps(data))
    )
    print(
        json.dumps(
            {
                "accepted": True,
                "verified_deadlines": 21,
                "viewer": str(args.out / "index.html"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
