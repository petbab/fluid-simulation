#!/usr/bin/env python3
"""
KTT dynamic-tuning result analyzer for SPH fluid simulation.

Consumes one or more KTT JSON result files (output of tuner.SaveResults)
and produces per-state optima, cross-state regret, kernel breakdown,
parameter sensitivity, and MCMC convergence figures.

Filename convention for state labelling:
    <state>_<anything>.json  ->  state = "settled" in "settled_run01.json"
Override with --labels label_map.json {"file.json": "state"}.

Optional sidecar CSV: <basename>.state.csv with columns
    timestamp,mean_speed,kinetic_energy,mean_neighbors,...
will be joined on nearest timestamp and exposed as state.* columns.

Usage:
    python analyze_ktt.py results/ --out figs/
    python analyze_ktt.py results/ --out figs/ --kernel simulation_step
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ loading --

def load_run(path: Path) -> pd.DataFrame:
    """Parse one KTT JSON file into a flat DataFrame, one row per Result."""
    data = json.loads(path.read_text())
    rows = []
    for r in data.get("Results", []):
        if r.get("Status") != "Ok":
            continue
        cfg = {p["Name"]: p["Value"] for p in r.get("Configuration", [])}
        kernels = {k["KernelFunction"]: k["Duration"]
                   for k in r.get("ComputationResults", [])}
        rows.append({
            "file": path.stem,
            "kernel_name": r.get("KernelName"),
            "timestamp": r.get("Timestamp"),
            "total_duration": r.get("TotalDuration"),
            "extra_duration": r.get("ExtraDuration", 0.0),
            "data_movement_overhead": r.get("DataMovementOverhead", 0.0),
            **{f"cfg.{k}": v for k, v in cfg.items()},
            **{f"k.{k}": v for k, v in kernels.items()},
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


def join_state_sidecar(df: pd.DataFrame, json_path: Path) -> pd.DataFrame:
    """Optional: join <basename>.state.csv on nearest timestamp."""
    sidecar = json_path.with_suffix(".state.csv")
    if not sidecar.exists() or df.empty:
        return df
    side = pd.read_csv(sidecar)
    if "timestamp" not in side.columns:
        return df
    side["timestamp"] = pd.to_datetime(side["timestamp"], utc=True)
    df = df.sort_values("timestamp")
    side = side.sort_values("timestamp")
    merged = pd.merge_asof(df, side, on="timestamp", direction="nearest",
                           tolerance=pd.Timedelta("2s"))
    rename = {c: f"state.{c}" for c in side.columns if c != "timestamp"}
    return merged.rename(columns=rename)


def load_dir(root: Path, label_map: dict[str, str] | None = None) -> pd.DataFrame:
    frames = []
    for p in sorted(root.rglob("*.json")):
        try:
            df = load_run(p)
        except Exception as e:
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        if df.empty:
            continue
        df = join_state_sidecar(df, p)
        state = (label_map or {}).get(p.name, p.stem.split("_")[0])
        df["state"] = state
        df["iter"] = np.arange(len(df))
        frames.append(df)
    if not frames:
        raise SystemExit(f"no usable KTT JSON files under {root}")
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------- helpers --

def cfg_cols(df: pd.DataFrame) -> list[str]:
    """All cfg.* columns that vary across the dataset (drops constants like KERNEL_DIR)."""
    return [c for c in df.columns
            if c.startswith("cfg.") and df[c].dropna().nunique() > 1]


def kernel_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("k.")]


def filter_kernel(df: pd.DataFrame, kernel_name: str | None) -> pd.DataFrame:
    if kernel_name is None:
        names = df["kernel_name"].dropna().unique()
        if len(names) != 1:
            raise SystemExit(f"multiple KernelNames {list(names)}; pass --kernel")
        return df
    return df[df["kernel_name"] == kernel_name]


# ----------------------------------------------------------------- analyses --

def best_per_state(df: pd.DataFrame) -> pd.DataFrame:
    cc = cfg_cols(df)
    g = (df.groupby(["state"] + cc, dropna=False)
         .agg(median_dur=("total_duration", "median"),
              p25=("total_duration", lambda x: x.quantile(0.25)),
              p75=("total_duration", lambda x: x.quantile(0.75)),
              n=("total_duration", "size"))
         .reset_index())
    return g.loc[g.groupby("state")["median_dur"].idxmin()].reset_index(drop=True)


def cross_state_regret(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """G[i,j] = (using i's best in j) / (j's own best). NaN if not measured."""
    cc = cfg_cols(df)
    median = (df.groupby(["state"] + cc, dropna=False)["total_duration"]
              .median().reset_index())
    states = sorted(median["state"].unique())
    best = {}
    for s in states:
        ms = median[median["state"] == s]
        row = ms.loc[ms["total_duration"].idxmin()]
        best[s] = (tuple(row[cc].tolist()), row["total_duration"])

    regret = pd.DataFrame(np.nan, index=states, columns=states, dtype=float)
    coverage = pd.DataFrame(False, index=states, columns=states)
    for tgt in states:
        own_best = best[tgt][1]
        pool = median[median["state"] == tgt].set_index(cc)
        for src in states:
            try:
                dur = pool.loc[best[src][0], "total_duration"]
                if isinstance(dur, pd.Series):
                    dur = float(dur.iloc[0])
                regret.loc[src, tgt] = dur / own_best
                coverage.loc[src, tgt] = True
            except KeyError:
                pass
    return regret, coverage


def kernel_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Mean per-kernel Duration at each state's best config."""
    bp = best_per_state(df)
    cc = cfg_cols(df)
    rows = []
    for _, r in bp.iterrows():
        mask = (df["state"] == r["state"])
        for c in cc:
            mask &= (df[c] == r[c]) | (df[c].isna() & pd.isna(r[c]))
        means = df[mask][kernel_cols(df)].mean()
        means["state"] = r["state"]
        rows.append(means)
    return pd.DataFrame(rows).set_index("state")


def parameter_marginals(df: pd.DataFrame, param: str) -> pd.DataFrame:
    return (df.groupby(["state", param])["total_duration"]
            .median().reset_index()
            .pivot(index=param, columns="state", values="total_duration"))


def convergence(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["state", "file", "iter"]).copy()
    out["best_so_far"] = (out.groupby(["state", "file"])["total_duration"]
                          .cummin())
    return out


# ------------------------------------------------------------------- plots --

def plot_kernel_breakdown(bk: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bk.plot(kind="bar", stacked=True, ax=ax, edgecolor="white", linewidth=0.4)
    ax.set_ylabel("mean duration per launch [ms]")
    ax.set_xlabel("fluid state")
    ax.set_title("Per-kernel cost at each state's optimum")
    ax.legend(title="kernel", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def plot_regret(regret: pd.DataFrame, coverage: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    arr = regret.values.astype(float)
    vmax = max(1.05, np.nanmax(arr)) if np.isfinite(arr).any() else 1.05
    im = ax.imshow(arr, cmap="rainbow_r", vmin=1.0, vmax=vmax)
    ax.set_xticks(range(len(regret.columns)))
    ax.set_xticklabels(regret.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(regret.index)))
    ax.set_yticklabels(regret.index)
    ax.set_xlabel("evaluated on state")
    ax.set_ylabel("config taken from state")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="white" if (not np.isnan(v) and v > 1.3) else "black")
    ax.set_title("Cross-state regret (× own best)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="× best")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def plot_parameter_marginals(df: pd.DataFrame, out: Path) -> None:
    cc = cfg_cols(df)
    if not cc:
        return
    n = len(cc); cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.2 * rows),
                             squeeze=False)
    states = sorted(df["state"].unique())
    cmap = plt.colormaps["tab10"]
    for k, p in enumerate(cc):
        ax = axes[k // cols][k % cols]
        marg = parameter_marginals(df, p)
        for i, s in enumerate(states):
            if s in marg.columns:
                marg[s].sort_index().plot(ax=ax, marker="o",
                                          color=cmap(i), label=s, linewidth=1.6)
        ax.set_title(p.removeprefix("cfg."), fontsize=10)
        ax.set_ylabel("median total [ms]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle("Parameter sensitivity per state", y=1.0, fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


def plot_convergence(df: pd.DataFrame, out: Path) -> None:
    conv = convergence(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    cmap = plt.colormaps["tab10"]
    for i, s in enumerate(sorted(conv["state"].unique())):
        sub = conv[conv["state"] == s]
        agg = (sub.groupby("iter")["best_so_far"]
               .agg(["median", "min", "max"]).reset_index())
        ax.plot(agg["iter"], agg["median"], color=cmap(i), label=s, linewidth=1.8)
        ax.fill_between(agg["iter"], agg["min"], agg["max"],
                        color=cmap(i), alpha=0.15)
    ax.set_xlabel("tuning iteration")
    ax.set_ylabel("best total duration so far [ms]")
    ax.set_title("MCMC convergence per state (median, [min, max] band)")
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)


# ------------------------------------------------------------------ report --

def write_report(df, bp, regret, coverage, bk, out: Path) -> None:
    cc = cfg_cols(df)
    L = ["# KTT dynamic-tuning analysis", "",
         f"- states: {sorted(df['state'].unique())}",
         f"- total result rows: {len(df)}",
         f"- varying parameters: {[c.removeprefix('cfg.') for c in cc]}",
         f"- runs per state: {df.groupby('state')['file'].nunique().to_dict()}",
         "",
         "## Best configuration per state", "",
         bp.drop(columns=['p25', 'p75'], errors='ignore').to_markdown(index=False),
         ""]
    if not regret.dropna(how="all").empty:
        L += ["## Cross-state regret (× own best)", "",
              regret.round(3).to_markdown(), "",
              "Coverage (config from row was actually measured in column):", "",
              coverage.to_markdown(), ""]
    L += ["## Per-kernel cost at each state's optimum [ms]", "",
          bk.round(4).to_markdown(), ""]
    out.write_text("\n".join(L))


# -------------------------------------------------------------------- main --

def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__))
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("figs"))
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--labels", type=Path, default=None)
    args = ap.parse_args()

    label_map = json.loads(args.labels.read_text()) if args.labels else None
    df = load_dir(args.results_dir, label_map)
    df = filter_kernel(df, args.kernel)
    if df.empty:
        raise SystemExit("no rows after filtering")

    args.out.mkdir(parents=True, exist_ok=True)
    bp = best_per_state(df)
    regret, coverage = cross_state_regret(df)
    bk = kernel_breakdown(df)

    plot_kernel_breakdown(bk, args.out / "01_kernel_breakdown.png")
    plot_regret(regret, coverage, args.out / "02_cross_state_regret.png")
    plot_parameter_marginals(df, args.out / "03_parameter_marginals.png")
    plot_convergence(df, args.out / "04_mcmc_convergence.png")

    df.to_csv(args.out / "tidy.csv", index=False)
    bp.to_csv(args.out / "best_per_state.csv", index=False)
    regret.to_csv(args.out / "cross_state_regret.csv")
    coverage.to_csv(args.out / "cross_state_regret_coverage.csv")
    bk.to_csv(args.out / "kernel_breakdown.csv")
    write_report(df, bp, regret, coverage, bk, args.out / "report.md")

    print(f"wrote {args.out}/  ({len(df)} rows, "
          f"{df['state'].nunique()} states, {df['file'].nunique()} files)")


if __name__ == "__main__":
    main()
