# Analysis Scripts

Python tooling for analyzing KTT autotuning results of the SPH fluid simulator.

## Setup

Requires Python ≥3.12. Dependencies are managed with [`uv`](https://docs.astral.sh/uv/):

```bash
cd analysis
uv sync
```

All scripts read from `../measurements/runs/` and write outputs to `./output/`.

## Experiments

| Experiment | Scripts | What it does |
|---|---|---|
| **E1** – Full tuning | [`e1_runtime_compile_dist.py`](e1_runtime_compile_dist.py), [`count_min_best.py`](count_min_best.py) | Plots runtime/compilation distributions and counts well-performing configurations per scenario. |
| **E2** – Cross-state transfer | [`e2_tables.py`](e2_tables.py), [`e2_visualize_tables.py`](e2_visualize_tables.py), [`e2_plot_step_times.py`](e2_plot_step_times.py) | Evaluates how a config tuned on one simulation state performs on another. Produces pivot tables, heatmaps, and step-time plots. |
| **E3** – Kernel composition | [`e3_distributions.py`](e3_distributions.py) | Stacked-bar visualization of median kernel runtimes per scenario. |
| **E4** – Cross-device comparison | [`e4_comparison.py`](e4_comparison.py), [`count_min_best_e4_avg.py`](count_min_best_e4_avg.py) | Compares median runtimes of configs tuned on RTX 3070 vs RTX 2080 Ti. |

## Utilities

- [`extract_config.py`](extract_config.py) – Extract the best configuration from a KTT result JSON.
- [`compare_configs.py`](compare_configs.py) – Side-by-side diff of two JSON config files.

## Running Experiments

Shell scripts in this directory automate data collection (run from `analysis/`):

- [`full-tuning.sh`](full-tuning.sh) – Run E1 full tuning for all states.
- [`cross-state.sh`](cross-state.sh) – Run E2 cross-state evaluation matrix.
- [`run-best-configs.sh`](run-best-configs.sh) – Run E3 validation with frozen best configs.

Example analysis workflow:

```bash
# E1: plot distributions
uv run e1-runtime-compile-dist

# E2: build tables and heatmaps
uv run e2-tables
uv run e2-visualize-tables

# E3: kernel composition plot
uv run e3-distributions

# E4: cross-device comparison
uv run e4-comparison
```
