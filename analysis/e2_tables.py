#!/usr/bin/env python3
"""
Read all measurements/runs/e2* files and produce two tables:
  - total runtime  (sum of TotalDuration per file)
  - mean frame time (mean of TotalDuration per file)

Rows = destinations (target states), columns = sources (source states).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_e2_filename(stem: str) -> tuple[str, str] | None:
    """e2_src-<src>_tgt-<tgt>  ->  (src, tgt)"""
    if not stem.startswith("e2_"):
        return None
    body = stem.removeprefix("e2_")
    # body looks like: src-BIG_falling_tgt-BIG_settled
    if "_tgt-" not in body:
        return None
    src_part, tgt_part = body.split("_tgt-", 1)
    src = src_part.removeprefix("src-").replace('_', '-')
    tgt = tgt_part.split("_rep", 1)[0].replace('_', '-')  # strip optional _repNN suffix
    return src, tgt


def load_e2_file(path: Path) -> dict | None:
    """Return {total_runtime_ms, mean_frame_time_ms, max_frame_time_ms, n_frames} for one e2 file."""
    data = json.loads(path.read_text())
    durations = []
    for r in data.get("Results", []):
        if r.get("Status") != "Ok":
            continue
        td = r.get("TotalDuration")
        if td is not None:
            durations.append(float(td))
    if not durations:
        return None
    arr = np.array(durations)
    return {
        "total_runtime_ms": arr.sum(),
        "mean_frame_time_ms": arr.mean(),
        "max_frame_time_ms": arr.max(),
        "n_frames": len(arr),
    }


def build_tables(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (total_runtime_table, mean_frame_time_table, max_frame_time_table, counts_table)."""
    records = []
    for path in sorted(runs_dir.glob("e2_*.json")):
        parsed = parse_e2_filename(path.stem)
        if parsed is None:
            continue
        src, tgt = parsed
        stats = load_e2_file(path)
        if stats is None:
            print(f"warning: no usable results in {path.name}", file=sys.stderr)
            continue
        records.append({
            "source": src,
            "target": tgt,
            **stats,
        })

    if not records:
        raise SystemExit(f"no usable e2 files in {runs_dir}")

    df = pd.DataFrame(records)

    # Pivot: rows = target, columns = source
    total_runtime = df.pivot(index="target", columns="source", values="total_runtime_ms")
    mean_frame = df.pivot(index="target", columns="source", values="mean_frame_time_ms")
    max_frame = df.pivot(index="target", columns="source", values="max_frame_time_ms")
    counts = df.pivot(index="target", columns="source", values="n_frames")

    # Ensure consistent ordering: sort rows and columns alphabetically
    total_runtime = total_runtime.sort_index().sort_index(axis=1)
    mean_frame = mean_frame.sort_index().sort_index(axis=1)
    max_frame = max_frame.sort_index().sort_index(axis=1)
    counts = counts.sort_index().sort_index(axis=1)

    return total_runtime, mean_frame, max_frame, counts


def normalize_by_diagonal(df: pd.DataFrame) -> pd.DataFrame:
    """Divide each row by its diagonal (source == target) value."""
    diag = pd.Series({idx: df.loc[idx, idx] for idx in df.index if idx in df.columns})
    return df.div(diag, axis=0)


def main() -> None:
    runs_dir = Path("../measurements/runs")
    if not runs_dir.exists():
        raise SystemExit(f"directory not found: {runs_dir}")

    total_runtime, mean_frame, max_frame, counts = build_tables(runs_dir)

    print("=" * 70)
    print("Total Runtime [ms]  (rows = destination, columns = source)")
    print("=" * 70)
    print(total_runtime.round(2).to_string())
    print()

    print("=" * 70)
    print("Mean Frame Time [ms]  (rows = destination, columns = source)")
    print("=" * 70)
    print(mean_frame.round(4).to_string())
    print()

    print("=" * 70)
    print("Max Frame Time [ms]  (rows = destination, columns = source)")
    print("=" * 70)
    print(max_frame.round(4).to_string())
    print()

    print("=" * 70)
    print("Normalized Total Runtime  (÷ source=destination)")
    print("=" * 70)
    norm_total = normalize_by_diagonal(total_runtime)
    print(norm_total.round(4).to_string())
    print()

    print("=" * 70)
    print("Normalized Mean Frame Time  (÷ source=destination)")
    print("=" * 70)
    norm_mean = normalize_by_diagonal(mean_frame)
    print(norm_mean.round(4).to_string())
    print()

    print("=" * 70)
    print("Normalized Max Frame Time  (÷ source=destination)")
    print("=" * 70)
    norm_max = normalize_by_diagonal(max_frame)
    print(norm_max.round(4).to_string())
    print()

    print("=" * 70)
    print("Frame Counts  (rows = destination, columns = source)")
    print("=" * 70)
    print(counts.to_string())
    print()

    # Also write CSVs for easy import elsewhere
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    total_runtime.round(2).to_csv(out_dir / "total_runtime.csv")
    mean_frame.round(4).to_csv(out_dir / "mean_frame_time.csv")
    max_frame.round(4).to_csv(out_dir / "max_frame_time.csv")
    norm_total.round(4).to_csv(out_dir / "total_runtime_normalized.csv")
    norm_mean.round(4).to_csv(out_dir / "mean_frame_time_normalized.csv")
    norm_max.round(4).to_csv(out_dir / "max_frame_time_normalized.csv")
    counts.to_csv(out_dir / "frame_counts.csv")
    print(f"Wrote CSVs to {out_dir}/")


if __name__ == "__main__":
    main()
