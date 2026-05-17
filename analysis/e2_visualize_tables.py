#!/usr/bin/env python3
"""Visualize output/ tables as colored heatmaps."""

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

OUT_DIR = Path("output")
PLOT_DIR = OUT_DIR


def load_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Load a CSV matrix and return (labels, data_array)."""
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        labels = header[1:]
        rows = []
        for row in reader:
            if not row:
                continue
            rows.append([float(v) for v in row[1:]])
    return labels, np.array(rows)


def plot_heatmap(labels: list[str], data: np.ndarray, title: str, out_path: Path) -> None:
    """Render a single heatmap and save it."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Choose colormap: diverging for normalized (centered at 1), sequential otherwise
    if "normalized" in title.lower():
        cmap = "RdYlGn_r"
        vmin, vmax = data.min(), data.max()
        # Center color scale around 1.0 if 1.0 is within range
        if vmin <= 1.0 <= vmax:
            max_dev = max(vmax - 1.0, 1.0 - vmin)
            vmin, vmax = 1.0 - max_dev, 1.0 + max_dev
    else:
        cmap = "viridis"
        vmin, vmax = data.min(), data.max()

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Labels
    ax.set_xlabel("Source", fontsize=11)
    ax.set_ylabel("Target", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("Value", rotation=270, labelpad=18)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = data[i, j]
            text_color = "white" if im.norm(val) < 0.4 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(OUT_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {OUT_DIR}")
        return

    for csv_path in csv_files:
        labels, data = load_csv(csv_path)
        title = csv_path.stem.replace("_", " ").title()
        out_path = PLOT_DIR / f"{csv_path.stem}.png"
        plot_heatmap(labels, data, title, out_path)

    print(f"\nAll plots written to {PLOT_DIR}")


if __name__ == "__main__":
    main()
