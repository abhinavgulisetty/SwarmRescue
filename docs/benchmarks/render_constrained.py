"""Render reproducible deadline-quality and exact-solve speed comparisons."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("report", type=Path)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
reader = gzip.open if args.report.suffix == ".gz" else open
with reader(args.report, "rt") as stream:
    report = json.load(stream)
methods = [
    "pareto",
    "astar",
    "lazy_astar",
    "dual_astar",
    "bounded_dp",
    "window_dp",
    "hybrid_exact",
    "milp",
]
labels = [
    "Legacy Pareto",
    "A*",
    "Lazy A*",
    "Dual A*",
    "Full-table DP",
    "Corridor DP",
    "Hybrid",
    "Native HiGHS",
]
colors = ["#8996a5"] * len(methods)
colors[6] = "#007d79"
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, (quality, speed) = plt.subplots(1, 2, figsize=(12, 5.4))
summary = report["summary"]
feasible = summary["hybrid_exact"]["feasible_cases"]
counts = [summary[m]["proven_optimal"] for m in methods]
quality.barh(labels, counts, color=colors)
quality.invert_yaxis()
quality.set_xlim(0, feasible * 1.16)
quality.set_xlabel(f"Proven optimal routes / {feasible} feasible cases")
for i, count in enumerate(counts):
    quality.text(count + feasible * 0.015, i, str(count), va="center")
ratios = [summary[m]["candidate_geomean_speedup_on_exact_pairs"] for m in methods]
limits = np.array([summary[m]["speedup_seed_bootstrap_95ci"] for m in methods])
errors = np.vstack(
    [np.maximum(0, ratios - limits[:, 0]), np.maximum(0, limits[:, 1] - ratios)]
)
speed.errorbar(
    ratios, np.arange(len(methods)), xerr=errors, fmt="o", color="#007d79", capsize=3
)
speed.set_yticks(np.arange(len(methods)), labels)
speed.invert_yaxis()
speed.set_xscale("log")
speed.axvline(1, color="#8996a5", linestyle="--", linewidth=1)
speed.set_xlabel("Hybrid speedup on mutually exact pairs (95% seed CI)")
fig.suptitle(
    f"Deadline routing: {len(report['cases'])} frozen matched cases", fontsize=15
)
fig.text(
    0.5,
    0.025,
    "Static estimated route objective; 200 KiB numeric arena. "
    "Host heaps / HiGHS RAM excluded. Author reference timing is reported separately.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.065, 1, 0.94))
args.out.parent.mkdir(parents=True, exist_ok=True)
for suffix in ("svg", "png"):
    fig.savefig(args.out.with_suffix("." + suffix), dpi=180)
