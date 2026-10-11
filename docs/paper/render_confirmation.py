"""Render the complete fresh comparison and its common-support sensitivity."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = {
    "native_exact": "Candidate",
    "native_astar": "Compiled lazy A*",
    "native_full_dp": "Compiled full DP",
    "native_window_dp": "Compiled corridor DP",
    "native_dp": "Compiled DP hybrid",
    "hybrid_exact": "Previous Python hybrid",
    "astar": "Python A*",
}


def main():
    report = json.loads((HERE / "confirmation-summary.json").read_text())
    matplotlib.rcParams.update({"font.size": 10, "svg.hashsalt": "ember-confirmation"})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    compiled = [m for m in report["statistics"] if m.startswith("native_")]
    bounds = []
    for offset, (source, label, color) in zip(
        (-0.12, 0.12),
        [
            ("statistics", "Pairwise exact support", "#245b82"),
            ("common_support_statistics", "All-method exact support", "#946037"),
        ],
        strict=True,
    ):
        for index, method in enumerate(compiled):
            result = report[source][method]
            value = result["candidate_geomean_speedup"]
            low, high = result["process_and_seed_bootstrap_95ci"]
            bounds.extend((low, high))
            axes[0].errorbar(
                value,
                index + offset,
                xerr=np.array([[value - low], [high - value]]),
                fmt="o",
                color=color,
                capsize=3,
                label=label if index == 0 else None,
            )
    axes[0].set_yticks(range(len(compiled)), [LABELS[m] for m in compiled])
    axes[0].invert_yaxis()
    axes[0].axvline(1, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_xlim(min(min(bounds), 1) - 0.03, max(max(bounds), 1) + 0.03)
    axes[0].set_xlabel("Baseline / candidate latency (larger favors candidate)")
    axes[0].set_title("Full-call speed ratios and approximate 95% intervals")
    axes[0].legend(loc="best", fontsize=8)
    methods = list(report["quality"])
    for index, method in enumerate(methods):
        quality = report["quality"][method]
        exact, total = quality["all_repeats_optimal"], quality["feasible_cases"]
        rate = 100 * exact / total
        axes[1].plot(rate, index, "o", color="#245b82")
        axes[1].text(rate - 2, index, f"{exact}/{total}", ha="right", va="center")
    axes[1].set_yticks(range(len(methods)), [LABELS[m] for m in methods])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 104)
    axes[1].set_xticks([0, 25, 50, 75, 100])
    axes[1].set_xlabel("Feasible cases proved optimal on all 21 repeats (%)")
    axes[1].set_title("Exact-solve coverage, including capped cases")
    for axis in axes:
        axis.grid(axis="x", color="#e5e5e5", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Preregistered confirmation: 240 maps, 720 cases, 3 processes, one host"
    )
    figure.savefig(HERE / "confirmation.svg", metadata={"Date": None})
    figure.savefig(HERE / "confirmation.png", dpi=180)
    plt.close(figure)
    with (HERE / "confirmation-table.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "method",
                "exact_pairs",
                "geomean_speed_ratio",
                "ci95_low",
                "ci95_high",
                "simultaneous_ci_low",
                "simultaneous_ci_high",
                "all_repeats_optimal",
                "feasible_cases",
                "worst_returned_gap_fraction",
            ]
        )
        for method, stats in report["statistics"].items():
            quality = report["quality"][method]
            writer.writerow(
                [
                    method,
                    stats["exact_pairs"],
                    stats["candidate_geomean_speedup"],
                    *stats["process_and_seed_bootstrap_95ci"],
                    *stats["bonferroni_simultaneous_95ci"],
                    quality["all_repeats_optimal"],
                    quality["feasible_cases"],
                    quality["worst_returned_gap_fraction"],
                ]
            )


if __name__ == "__main__":
    main()
