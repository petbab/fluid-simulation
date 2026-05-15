#!/usr/bin/env python3
"""
For measurements/runs/e1* files, computes:
  - scenario name (file name stripped of e1_ and _rep00.json)
  - best runtime (minimum within first 1686 of the e1 file)
  - matching runtime from e1 that corresponds to the e4 configuration
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
    if basename.endswith("_rep00.json"):
        basename = basename[:-11]
    elif basename.endswith(".json"):
        basename = basename[:-5]
    return basename.replace('_', '-')


def get_e4_path(e1_path: str) -> str:
    """Derive the corresponding e4 file path from an e1 file path."""
    dirname = os.path.dirname(e1_path)
    basename = os.path.basename(e1_path)
    # e1_<scenario>_rep00.json -> e4_<scenario>.json
    if basename.startswith("e1_") and basename.endswith("_rep00.json"):
        scenario = basename[3:-11]
        return os.path.join(dirname, f"e4_{scenario}.json")
    raise ValueError(f"Unexpected e1 filename format: {basename}")


def load_numeric_config(result: dict) -> tuple:
    """Extract only UnsignedInt/Double parameters as a sorted tuple for hashing."""
    return tuple(sorted(
        (entry["Name"], entry["Value"])
        for entry in result["Configuration"]
        if entry.get("ValueType") in ("UnsignedInt", "Double")
    ))


def analyze_file(e1_path: str, eps: float) -> dict:
    with open(e1_path) as f:
        e1_data = json.load(f)

    e4_path = get_e4_path(e1_path)
    with open(e4_path) as f:
        e4_data = json.load(f)

    e1_results = e1_data["Results"]
    e1_subset = e1_results[:1686]
    e1_durations = [r["TotalDuration"] for r in e1_subset]

    # Build lookup from numeric config -> list of durations in e1
    e1_lookup = {}
    for r in e1_results:
        cfg = load_numeric_config(r)
        dur = r["TotalDuration"]
        if cfg not in e1_lookup:
            e1_lookup[cfg] = []
        e1_lookup[cfg].append(dur)

    # Get the (first) numeric config from e4
    e4_cfg = load_numeric_config(e4_data["Results"][0])

    # Look up the matching config in e1
    if e4_cfg in e1_lookup:
        matching_durations = e1_lookup[e4_cfg]
        matching_avg = sum(matching_durations) / len(matching_durations)
    else:
        raise ValueError(
            f"e4 config from {e4_path} not found in {e1_path}. "
            f"Config: {dict(e4_cfg)}"
        )

    best = min(e1_durations)
    threshold = (1.0 + eps) * best
    count = sum(1 for d in e1_durations if d < threshold)
    percentage = 100.0 * count / len(e1_durations)
    avg = sum(e1_durations) / len(e1_durations)

    trials = len(e1_durations) / count
    min_steps = math.ceil(
        trials * (COMPILATION_TIME + avg - threshold) / (matching_avg - threshold)
    )

    return {
        "scenario": extract_name(e1_path),
        "best": best,
        "matching_avg": matching_avg,
        "percentage": percentage,
        "count": count,
        "total": len(e1_durations),
        "min_steps": min_steps,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python count_min_best_e4_avg.py <eps> [file_pattern] [output.csv]")
        print("  eps         : float, e.g. 0.1 for 10%")
        print("  file_pattern: glob pattern for e1 files (default: measurements/runs/e1*)")
        print("  output.csv  : output CSV path (default: analysis/min_steps_e4.csv)")
        sys.exit(1)

    eps = float(sys.argv[1])
    pattern = sys.argv[2] if len(sys.argv) > 2 else "../measurements/runs/e1*"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "min_steps_e4.csv"

    files = glob.glob(pattern)
    if not files:
        print(f"No files matched pattern: {pattern}")
        sys.exit(1)

    rows = []
    for fp in sorted(files):
        try:
            rows.append(analyze_file(fp, eps))
        except:
            print(fp)

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Scenario", "Best", "MatchingAvg", "Well-performing", "Min steps"])
        for row in rows:
            writer.writerow([
                row["scenario"],
                f"{row['best']:.3f}",
                f"{row['matching_avg']:.3f}",
                f"{row['count']}",
                f"{row['min_steps']}",
            ])

    print(f"Saved table to {out_path}")


if __name__ == "__main__":
    main()
