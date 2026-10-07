"""Render the committed frozen measurements without rerunning evaluation."""

import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    folder = Path(__file__).resolve().parent
    with gzip.open(folder / "solver-evaluation.json.gz", "rt") as source:
        report = json.load(source)
    families = [None, *report["families"]]
    labels = [
        "All 144 models",
        "Directed corridors",
        "Cyclic grids",
        "Sticky self-loops",
        "Generated rooms",
        "Dense graphs",
        "Disconnected grids",
    ]
    plt.rcParams.update({"font.size": 11, "svg.hashsalt": report["protocol_sha256"]})
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    for method, label, color, offset in (
        ("jacobi", "Compared with compiled Jacobi", "#2679a7", -0.13),
        ("gauss_seidel", "Compared with Gauss-Seidel", "#c27724", 0.13),
    ):
        for position, family in enumerate(families):
            rows = [
                row
                for row in report["cases"]
                if family is None or row["family"] == family
            ]
            clusters = []
            for seed in report["seeds"]:
                selected = [row for row in rows if row["seed"] == seed]
                logs = [
                    np.log(
                        row["results"][method]["end_to_end"]["median_ms"]
                        / row["results"]["auto"]["end_to_end"]["median_ms"]
                    )
                    for row in selected
                ]
                clusters.append(np.mean(logs))
            mean = float(np.exp(np.mean(clusters)))
            bootstrap = (
                np.random.default_rng(49381)
                .choice(clusters, size=(2000, len(clusters)), replace=True)
                .mean(axis=1)
            )
            low, high = np.exp(np.quantile(bootstrap, [0.025, 0.975]))
            ax.errorbar(
                mean,
                position + offset,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                color=color,
                capsize=3,
                label=label if position == 0 else None,
            )
            ax.text(
                high * 1.09,
                position + offset,
                f"{mean:.2f}x",
                va="center",
                fontsize=10,
                color=color,
            )
    ax.axvline(1, color="#666666", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(0.72, 31)
    ax.set_xticks(
        [0.8, 1, 2, 5, 10, 20], labels=["0.8x", "1x", "2x", "5x", "10x", "20x"]
    )
    ax.set_yticks(range(len(families)), labels=labels)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#dddddd", linewidth=0.5)
    ax.set_xlabel("Baseline solve time / adaptive solve time  •  right of 1x is faster")
    ax.set_title(
        "Faster routing solves with the same integer field", loc="left", pad=18
    )
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.01,
        0.012,
        "Frozen 144-model evaluation • 24 seeds • "
        "95% seed-cluster bootstrap intervals\n"
        "Quantization, allocation and preprocessing included • host C99 -O2 • "
        "rescue quality and hardware cycles not measured",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(folder / "solver-speedups.png", dpi=160)
    fig.savefig(folder / "solver-speedups.svg")


if __name__ == "__main__":
    main()
