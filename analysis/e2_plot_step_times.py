#!/usr/bin/env python3
"""Plot simulation step times for given target and list of sources."""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

RUNS_DIR = Path(__file__).parent.parent / "measurements" / "runs"
PLOT_DIR = Path(__file__).with_suffix("").parent / "out" / "plots"

# Add vertical lines at specific steps with annotations.
# Each entry is a tuple: (step_index, annotation_text)
VERTICAL_LINES: list[tuple[int, str]] = [
    # (0, "Snapshot 1"),
    # (350, "Snapshot 2"),
    # (700, "Snapshot 3"),
]


def load_step_times(path: Path) -> list[float]:
    """Load TotalDuration values from a KTT results JSON file."""
    with path.open() as f:
        data = json.load(f)
    return [r["TotalDuration"] for r in data["Results"]]


def plot_step_times(tgt: str, srcs: list[str], out_path: Path) -> None:
    """Plot step times for multiple sources targeting the same tgt."""
    fig, ax = plt.subplots(figsize=(12, 6))

    if tgt not in srcs:
        srcs.append(tgt)

    for src in srcs:
        file_path = RUNS_DIR / f"e2_src-{src}_tgt-{tgt}.json"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping.")
            continue
        step_times = load_step_times(file_path)
        ax.plot(step_times, label=src, linewidth=0.8)

    for step, annotation in VERTICAL_LINES:
        ax.axvline(x=step, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(
            step,
            ax.get_ylim()[1],
            annotation,
            color="red",
            fontsize=10,
            ha="right",
            va="top",
            rotation=90,
        )

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Total Duration (ms)", fontsize=11)
    ax.set_title(f"Simulation Step Times — Target: {tgt}", fontsize=13)
    ax.legend(title="Source", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot simulation step times for a target and list of sources."
    )
    parser.add_argument("tgt", help="Target scenario name")
    parser.add_argument(
        "srcs", nargs="*", help="One or more source scenario names"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: analysis/out/plots/e2_step_times_<tgt>.png)",
    )
    args = parser.parse_args()

    out_path = args.output or PLOT_DIR / f"e2_step_times_{args.tgt}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_step_times(args.tgt, args.srcs or [args.tgt], out_path)


if __name__ == "__main__":
    main()
