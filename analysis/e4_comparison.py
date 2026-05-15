#!/usr/bin/env python3
"""Compare median total runtimes between e3 and e4 results per scenario."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_median_runtimes(runs_dir: Path, prefix: str):
    """Load all {prefix}_*.json files and return median TotalDuration per scenario."""
    files = sorted(runs_dir.glob(f"{prefix}_*.json"))
    if not files:
        raise FileNotFoundError(f"No {prefix}_*.json files found in {runs_dir}")

    medians = {}
    for filepath in files:
        with open(filepath, "r") as f:
            doc = json.load(f)

        total_durations = []
        for result in doc.get("Results", []):
            total = result.get("TotalDuration")
            if total is not None:
                total_durations.append(float(total))

        if total_durations:
            scenario = filepath.stem.replace(f"{prefix}_", "")
            medians[scenario] = np.median(total_durations)

    return medians


def plot_comparison(e3_medians, e4_medians, output_path: Path | None = None):
    """Plot a grouped bar chart comparing e3 and e4 median runtimes."""
    scenarios = sorted(set(e3_medians.keys()) & set(e4_medians.keys()))
    if not scenarios:
        print("No common scenarios found between e3 and e4.")
        sys.exit(1)

    e3_values = [e3_medians[s] for s in scenarios]
    e4_values = [e4_medians[s] for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.2), 6))
    bars1 = ax.bar(x - width / 2, e3_values, width, label="Tuned on 3070", color="steelblue", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, e4_values, width, label="Tuned on 2080Ti", color="coral", edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Median Total Duration (ms)")
    ax.set_title("Median Runtime Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bars with values
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    runs_dir = Path(__file__).parent.parent / "measurements" / "runs"
    if not runs_dir.exists():
        print(f"Directory not found: {runs_dir}")
        sys.exit(1)

    e3_medians = load_median_runtimes(runs_dir, "e3")
    e4_medians = load_median_runtimes(runs_dir, "e4")

    print("e3 median runtimes (ms):")
    for s, v in sorted(e3_medians.items()):
        print(f"  {s}: {v:.3f}")
    print("e4 median runtimes (ms):")
    for s, v in sorted(e4_medians.items()):
        print(f"  {s}: {v:.3f}")

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    plot_comparison(e3_medians, e4_medians, out_dir / "e4_runtime_comparison.png")


if __name__ == "__main__":
    main()
