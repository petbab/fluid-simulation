#!/usr/bin/env python3
"""Parse KTT e3 results and visualize kernel runtime composition per scenario."""

import json
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_e3_data(runs_dir: Path):
    """Load all e3_*.json files and return median kernel durations per file."""
    files = sorted(runs_dir.glob("e3_*.json"))
    if not files:
        raise FileNotFoundError(f"No e3_*.json files found in {runs_dir}")

    # file_name -> {kernel: [durations]}
    raw = {}
    for filepath in files:
        with open(filepath, "r") as f:
            doc = json.load(f)

        file_data = {}
        for result in doc.get("Results", []):
            for comp in result.get("ComputationResults", []):
                kernel = comp.get("KernelFunction")
                duration = comp.get("Duration")
                if kernel is None or duration is None:
                    continue
                file_data.setdefault(kernel, []).append(float(duration))
        raw[filepath.name] = file_data

    # Compute median per kernel per file
    file_medians = {}
    all_kernels = set()
    for fname, kdata in raw.items():
        medians = {}
        for kernel, durations in kdata.items():
            medians[kernel] = np.median(durations)
            all_kernels.add(kernel)
        file_medians[fname] = medians

    return file_medians, sorted(all_kernels), files


def plot_stacked_bars(file_medians, all_kernels, output_path: Path | None = None):
    """Plot a single stacked bar per file showing median kernel durations."""
    files = sorted(file_medians.keys())
    n_files = len(files)
    n_kernels = len(all_kernels)

    # Consistent color per kernel
    cmap = plt.cm.get_cmap("tab20", n_kernels)
    colors = {k: cmap(i) for i, k in enumerate(all_kernels)}

    fig, ax = plt.subplots(figsize=(max(8, n_files * 1.2), 6))

    x = np.arange(n_files)
    bottom = np.zeros(n_files)

    for kernel in all_kernels:
        values = np.array([file_medians[f].get(kernel, 0.0) for f in files])
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=kernel,
            color=colors[kernel],
            width=0.6,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += values

    # Labels
    labels = [f.replace("e3_", "").replace(".json", "") for f in files]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Median Duration (ms)")
    ax.set_title("Kernel Runtime Composition per Scenario (KTT Results)")

    # Total height annotation
    totals = [
        sum(file_medians[f].get(k, 0.0) for k in all_kernels) for f in files
    ]
    for i, total in enumerate(totals):
        ax.annotate(
            f"{total:.2f}",
            xy=(i, total),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.legend(
        title="Kernel",
        # bbox_to_anchor=(1.02, 1),
        loc="upper right",
        fontsize="small",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
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

    file_medians, all_kernels, files = load_e3_data(runs_dir)
    print(f"Loaded {len(files)} e3 files, kernels: {all_kernels}")

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    plot_stacked_bars(
        file_medians, all_kernels, out_dir / "e3_kernel_runtime_composition.png"
    )


if __name__ == "__main__":
    main()
