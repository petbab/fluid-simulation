#!/usr/bin/env python3
"""
For measurements/runs/e1* files, computes:
  - scenario name (file name stripped of e1_ and .json)
  - best runtime (minimum within first 1686)
  - average runtime (mean of all results)
  - percentage of results within a given eps threshold of the best

Outputs a CSV file.
"""

import csv
import json
import sys
import glob
import os
import math

COMPILATION_TIME = 870.82

def extract_name(filepath: str) -> str:
    basename = os.path.basename(filepath)
    if basename.startswith("e1_"):
        basename = basename[3:]
    elif basename.endswith(".json"):
        basename = basename[:-5]
    return basename.replace('_', '-')


def analyze_file(filepath: str, eps: float) -> dict:
    with open(filepath) as f:
        data = json.load(f)

    results = data["Results"]
    subset = results[:1686]
    durations = [r["TotalDuration"] for r in subset]

    best = min(durations)
    avg = sum(durations) / len(durations)
    threshold = (1.0 + eps) * best
    count = sum(1 for d in durations if d < threshold)
    percentage = 100.0 * count / len(durations)

    trials = len(durations) / count
    min_steps = math.ceil(trials * (COMPILATION_TIME + avg - threshold) / (avg - threshold))

    return {
        "scenario": extract_name(filepath),
        "best": best,
        "average": avg,
        "percentage": percentage,
        "count": count,
        "total": len(durations),
        "min_steps": min_steps,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python count_min_steps.py <eps> [file_pattern] [output.csv]")
        print("  eps         : float, e.g. 0.1 for 10%")
        print("  file_pattern: glob pattern for e1 files (default: ../measurements/runs/e1*)")
        print("  output.csv  : output CSV path (default: output/min_steps.csv)")
        sys.exit(1)

    eps = float(sys.argv[1])
    pattern = sys.argv[2] if len(sys.argv) > 2 else "../measurements/runs/e1*"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "output/min_steps.csv"

    files = glob.glob(pattern)
    if not files:
        print(f"No files matched pattern: {pattern}")
        sys.exit(1)

    rows = [analyze_file(fp, eps) for fp in sorted(files)]

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Scenario", "Best", "Average", "Well-performing", "Min steps"])
        for row in rows:
            writer.writerow([
                row["scenario"],
                f"{row['best']:.3f}",
                f"{row['average']:.3f}",
                f"{row['count']}",
                f"{row['min_steps']}",
            ])

    print(f"Saved table to {out_path}")


if __name__ == "__main__":
    main()
