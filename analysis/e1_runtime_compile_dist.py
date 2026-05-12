#!/usr/bin/env python3
"""
For each measurements/runs/e1*.json file, plot the distribution of run times
and compilation times for the first 1686 measurements.

Run time per measurement  = Result.TotalDuration
Compile time per measurement = sum of CompilationOverhead across all kernels
in Result.ComputationResults.
"""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def percentile(data, p):
    """Return the p-th percentile (0-100) using linear interpolation."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    if f == c:
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def summarize(name, values):
    print(f"\n{name}")
    print("-" * 40)
    print(f"  count   : {len(values)}")
    print(f"  min     : {min(values):.6f} ms")
    print(f"  max     : {max(values):.6f} ms")
    print(f"  mean    : {statistics.mean(values):.6f} ms")
    print(f"  median  : {statistics.median(values):.6f} ms")
    print(f"  stddev  : {statistics.stdev(values):.6f} ms")
    for p in [5, 25, 50, 75, 95, 99]:
        print(f"  p{p:02d}     : {percentile(values, p):.6f} ms")


def plot_distributions(files_data, output_dir: Path):
    """Plot run-time and compile-time distributions for all files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n_files = len(files_data)
    cols = 3
    rows = (n_files + cols - 1) // cols

    # --- Run time distributions ---
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    for idx, (fname, run_times, _) in enumerate(files_data):
        ax = axes[idx // cols, idx % cols]
        ax.hist(run_times, bins=50, color="steelblue", edgecolor="white")
        ax.set_title(fname.replace("e1_", "").replace(".json", ""))
        ax.set_xlabel("Run time (ms)")
        ax.set_ylabel("Count")
        ax.axvline(statistics.median(run_times), color="red", linestyle="--", label="median")
        ax.legend()
    # Hide unused subplots
    for idx in range(n_files, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)
    fig.suptitle("Run Time Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    run_plot_path = output_dir / "e1_run_time_distributions.png"
    fig.savefig(run_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved run-time plot: {run_plot_path}")

    # --- Compilation time distributions ---
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
    for idx, (fname, _, compile_times) in enumerate(files_data):
        ax = axes[idx // cols, idx % cols]
        ax.hist(compile_times, bins=50, color="darkorange", edgecolor="white")
        ax.set_title(fname.replace("e1_", "").replace(".json", ""))
        ax.set_xlabel("Compilation time (ms)")
        ax.set_ylabel("Count")
        ax.axvline(statistics.median(compile_times), color="red", linestyle="--", label="median")
        ax.legend()
    for idx in range(n_files, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)
    fig.suptitle("Compilation Time Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    comp_plot_path = output_dir / "e1_compile_time_distributions.png"
    fig.savefig(comp_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved compile-time plot: {comp_plot_path}")


def main():
    runs_dir = Path("../measurements/runs")
    files = sorted(runs_dir.glob("e1*.json"))
    if not files:
        raise SystemExit(f"No e1*.json files found in {runs_dir}")

    n = 1686
    files_data = []

    for filepath in files:
        print(f"\nProcessing: {filepath.name}")
        with open(filepath, "r") as f:
            data = json.load(f)

        results = data.get("Results", [])
        if len(results) < n:
            print(f"  Skipped (only {len(results)} results, need {n})")
            continue

        run_times = []
        compile_times = []

        for r in results[:n]:
            run_times.append(float(r.get("TotalDuration", 0.0)))
            comp_overheads = [
                float(k.get("CompilationOverhead", 0.0))
                for k in r.get("ComputationResults", [])
            ]
            compile_times.append(sum(comp_overheads))

        summarize("  Run time per measurement (TotalDuration)", run_times)
        summarize("  Compilation time per measurement (sum of CompilationOverhead)", compile_times)
        files_data.append((filepath.name, run_times, compile_times))

    if files_data:
        plot_distributions(files_data, Path("out"))


if __name__ == "__main__":
    main()
