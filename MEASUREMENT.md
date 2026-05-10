# Agent measurement runbook

Execute in order. After each phase, verify the listed artifacts exist before continuing. Every command runs from the repo root unless stated otherwise.

## 0 · Environment

```bash
# Lock GPU clocks (run as root or with sudo).
nvidia-smi -pm 1
nvidia-smi --query-supported-clocks=gr --format=csv,noheader,nounits | head
# Pick a base graphics clock from the list, then:
nvidia-smi -lgc <base_clock>          # e.g. 1500
# Confirm:
nvidia-smi --query-gpu=clocks.gr,clocks.mem,persistence_mode --format=csv

# Disable any other GPU users (no other CUDA/OpenGL apps, no display compositor on the same GPU if possible).
```

Build a Release binary tree:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Verify the six binaries exist under `build/bin/`: `BIG`, `dragon_collision`, `fountain`, `surface_tension`, `tilting_box`, `vortex`. **If any fails to build, stop and report.**

Create the workspace:

```bash
mkdir -p measurements/{snapshots,configs,runs,figs}
```

## 1 · Scenario roster

These are the fluid-state buckets, mapped to existing apps:

| label              | binary             | regime characteristic                 |
|--------------------|--------------------|---------------------------------------|
| `dragon_settled`   | `dragon_collision` | settled, dense, low velocity          |
| `dragon_falling`   | `dragon_collision` | freshly released, sparse, falling     |
| `fountain_active`  | `fountain`         | persistent forcing, mixed densities   |
| `vortex_active`    | `vortex`           | swirling, stable structure            |
| `tilting_chaotic`  | `tilting_box`      | non-stationary boundary, sloshing     |
| `surface_settled`  | `surface_tension`  | stable cluster, surface forces        |

Six states is enough for a 6×6 regret matrix. If any binary fails to run for ≥1 step on its own snapshot, drop it from the roster and continue with the rest.

## 2 · Snapshot creation

Each scenario needs one **settled** snapshot (used by E1, E2, parts of E4) and the `dragon_collision` scenario also needs a **falling** snapshot (its trajectory is the E4 transient).

For each state where the regime is "the natural state of the simulation after letting it run", produce the snapshot headlessly:

```bash
# settled snapshots — drop fluid, simulate to equilibrium, save
./build/bin/dragon_collision --headless --warmup-iters 0 --fixed-dt 0.01 \
    --stop sim-time=5.0 --snapshot-save measurements/snapshots/dragon_settled.sphs

./build/bin/fountain --headless --warmup-iters 0 --fixed-dt 0.01 \
    --stop sim-time=5.0 --snapshot-save measurements/snapshots/fountain_active.sphs

./build/bin/vortex --headless --warmup-iters 0 --fixed-dt 0.01 \
    --stop sim-time=5.0 --snapshot-save measurements/snapshots/vortex_active.sphs

./build/bin/tilting_box --headless --warmup-iters 0 --fixed-dt 0.01 \
    --stop sim-time=5.0 --snapshot-save measurements/snapshots/tilting_chaotic.sphs

./build/bin/surface_tension --headless --warmup-iters 0 --fixed-dt 0.01 \
    --stop sim-time=5.0 --snapshot-save measurements/snapshots/surface_settled.sphs

# falling snapshot for dragon — taken early, before fluid hits the obstacle
./build/bin/dragon_collision --headless --warmup-iters 0 --fixed-dt 0.01 \
    --stop sim-time=0.3 --snapshot-save measurements/snapshots/dragon_falling.sphs
```

Verify each `.sphs` is non-empty and the binary exited 0. If `surface_tension` or `tilting_box` produces a stable settled state in <5 s, the longer simulation just continues at equilibrium — fine.

`dragon_falling` becomes the seventh state for E1/E2.

## 3 · E1 — per-state best configuration

Three replicates per state, MCMC with iteration budget of 200. Each invocation writes its own JSON via the existing `Tuner::~Tuner` save path — pin the output prefix per run.

```bash
STATES=(dragon_settled dragon_falling fountain_active vortex_active \
        tilting_chaotic surface_settled)
declare -A BIN=(
  [dragon_settled]=dragon_collision
  [dragon_falling]=dragon_collision
  [fountain_active]=fountain
  [vortex_active]=vortex
  [tilting_chaotic]=tilting_box
  [surface_settled]=surface_tension
)

for s in "${STATES[@]}"; do
  for r in 0 1 2; do
    out="measurements/runs/e1_${s}_rep$(printf %02d $r)"
    ./build/bin/${BIN[$s]} --headless \
      --snapshot-load measurements/snapshots/${s}.sphs \
      --searcher mcmc --tuning-budget 1.0 \
      --warmup-iters 200 --fixed-dt 0.01 \
      --stop iters=200 \
      --ktt-output "${out}" \
      --seed $r
    sleep 20   # GPU cooldown
  done
done
```

After this completes, every `measurements/runs/e1_<state>_rep??.json` should exist. Spot-check one with `python -c 'import json; print(len(json.load(open("measurements/runs/e1_dragon_settled_rep00.json"))["Results"]))'` — expect ~200.

## 4 · Extract the per-state best configs (for E2 and E4)

Build a tiny extractor (write it once, reuse):

```bash
cat > tools/extract_config.py <<'EOF'
#!/usr/bin/env python3
"""extract_config.py result.json -> config.json (best by TotalDuration)."""
import json, sys
data = json.loads(open(sys.argv[1]).read())
ok = [r for r in data["Results"] if r.get("Status") == "Ok"]
best = min(ok, key=lambda r: r["TotalDuration"])
# Strip parameters that are environment-specific or fixed; keep tunable ones.
DROP = {"KERNEL_DIR", "EXTERNAL_FORCE"}
cfg = [p for p in best["Configuration"] if p["Name"] not in DROP]
json.dump(cfg, open(sys.argv[2], "w"), indent=2)
print(f"{sys.argv[1]} -> {sys.argv[2]}  ({best['TotalDuration']:.4f} ms)")
EOF
chmod +x tools/extract_config.py

# Pick the best replicate (lowest min duration) per state and extract.
for s in "${STATES[@]}"; do
  best_run=$(python3 -c "
import json, glob
files = sorted(glob.glob(f'measurements/runs/e1_${s}_rep*.json'))
mins = [(min(r['TotalDuration'] for r in json.load(open(f))['Results'] if r.get('Status')=='Ok'), f) for f in files]
print(min(mins)[1])")
  python3 tools/extract_config.py "$best_run" "measurements/configs/${s}.json"
done

# Empty default config = fall back to KTT's first-listed values for every parameter
echo '[]' > measurements/configs/default.json
```

Verify `measurements/configs/` contains 6 state-specific JSONs plus `default.json`.

## 5 · E2 — cross-state regret

For each (source config, target state) pair, run **frozen** with that config in that state for 100 measured iterations. Three replicates.

```bash
for tgt in "${STATES[@]}"; do
  for src in "${STATES[@]}" default; do
    for r in 0 1 2; do
      out="measurements/runs/e2_src-${src}_tgt-${tgt}_rep$(printf %02d $r)"
      ./build/bin/${BIN[$tgt]} --headless \
        --snapshot-load measurements/snapshots/${tgt}.sphs \
        --frozen-config measurements/configs/${src}.json \
        --tuning-budget 0.0 \
        --warmup-iters 200 --fixed-dt 0.01 \
        --stop iters=100 \
        --ktt-output "${out}" \
        --seed $r
      sleep 5
    done
  done
done
```

That's 7 sources × 6 targets × 3 reps = 126 runs at ~20 s each ≈ 45 min. Spot-check one resulting JSON: every Result should have the same `Configuration` (the source config's parameters) and 100 entries.

## 6 · E3 — convergence (no extra measurements)

Reuse E1 logs. The analyzer (delivered earlier) computes `best_so_far` over MCMC iterations from the existing JSONs. No commands required here.

## 7 · E4 — end-to-end on the dragon-collision transient

The dragon scenario is the natural transient: fluid block released → falls → impacts the dragon and the box → settles. Run from `dragon_falling` snapshot for ~10 s of sim time. Compare four variants:

```bash
declare -A E4_CONFIG=(
  [B0]=measurements/configs/default.json
  [B1]=measurements/configs/dragon_settled.json     # tuned once on settled
  [B2]=measurements/configs/dragon_falling.json     # oracle for this transient's start
  # T1 has no frozen config — dynamic tuning is on
)

for variant in B0 B1 B2 T1; do
  for r in 0 1 2 3 4; do
    out="measurements/runs/e4_${variant}_rep$(printf %02d $r)"
    if [ "$variant" = "T1" ]; then
      ./build/bin/dragon_collision --headless \
        --snapshot-load measurements/snapshots/dragon_falling.sphs \
        --searcher mcmc --tuning-budget 0.1 \
        --warmup-iters 200 --fixed-dt 0.01 \
        --stop sim-time=10.0 \
        --log-csv "${out}.steps.csv" --log-metrics \
        --ktt-output "${out}" \
        --seed $r
    else
      ./build/bin/dragon_collision --headless \
        --snapshot-load measurements/snapshots/dragon_falling.sphs \
        --frozen-config "${E4_CONFIG[$variant]}" \
        --tuning-budget 0.0 \
        --warmup-iters 200 --fixed-dt 0.01 \
        --stop sim-time=10.0 \
        --log-csv "${out}.steps.csv" --log-metrics \
        --ktt-output "${out}" \
        --seed $r
    fi
    sleep 20
  done
done
```

Randomize the variant order between replicates if you want extra robustness — replace the outer `variant` loop with a shuffled list for each `r`.

Per-step CSVs are the primary E4 data; KTT JSONs are kept as backup.

## 8 · E5 — tuning-budget sweep

Same dragon-collision transient, T1 only, sweeping the budget:

```bash
for tb in 0.0 0.05 0.1 0.25 0.5 1.0; do
  for r in 0 1 2; do
    tag=$(echo $tb | tr . _)
    out="measurements/runs/e5_tb-${tag}_rep$(printf %02d $r)"
    ./build/bin/dragon_collision --headless \
      --snapshot-load measurements/snapshots/dragon_falling.sphs \
      --searcher mcmc --tuning-budget $tb \
      --warmup-iters 200 --fixed-dt 0.01 \
      --stop sim-time=10.0 \
      --log-csv "${out}.steps.csv" \
      --ktt-output "${out}" \
      --seed $r
    sleep 20
  done
done
```

## 9 · Analysis

E1, E2, E3 — feed all the JSONs into the analyzer:

```bash
# Build a labels file mapping E1 filenames -> state
python3 - <<'EOF' > measurements/labels.json
import json, glob, os
labels = {}
for f in glob.glob("measurements/runs/e1_*.json"):
    state = os.path.basename(f).removeprefix("e1_").rsplit("_rep",1)[0]
    labels[os.path.basename(f)] = state
for f in glob.glob("measurements/runs/e2_*.json"):
    # e2_src-X_tgt-Y_repNN.json — label by target (the regret matrix's column = state we ran in)
    name = os.path.basename(f).removeprefix("e2_")
    src = name.split("_tgt-")[0].removeprefix("src-")
    tgt = name.split("_tgt-")[1].split("_rep")[0]
    # Tag both source and target so analyzer can build the matrix; here we only need target.
    labels[os.path.basename(f)] = tgt
json.dump(labels, open("/dev/stdout","w"), indent=2)
EOF

python3 analyze_ktt.py measurements/runs --out measurements/figs \
    --labels measurements/labels.json
```

The analyzer's regret matrix needs both axes to know each row's source. Since file names already encode it, extend the analyzer with a tiny tweak before running — or, simpler, run E2 separately:

```bash
mkdir -p measurements/runs_e1
cp measurements/runs/e1_*.json measurements/runs_e1/
python3 analyze_ktt.py measurements/runs_e1 --out measurements/figs/e1_e3
```

For E2, build the regret matrix directly (don't rely on the analyzer's MCMC-shaped logic):

```bash
cat > tools/build_regret.py <<'EOF'
#!/usr/bin/env python3
import json, glob, os, re, statistics, csv, pathlib
runs = {}
for f in glob.glob("measurements/runs/e2_*.json"):
    m = re.match(r"e2_src-(.+)_tgt-(.+)_rep(\d+)\.json", os.path.basename(f))
    src, tgt, _ = m.groups()
    medians = sorted(r["TotalDuration"] for r in json.load(open(f))["Results"] if r.get("Status")=="Ok")
    runs.setdefault((src,tgt), []).append(statistics.median(medians))

states = sorted({s for s,_ in runs.keys()} | {t for _,t in runs.keys()})
own_best = {t: min(statistics.median(v) for (s,tt),v in runs.items() if tt==t and s==t)
            for t in states if (t,t) in runs}

out = pathlib.Path("measurements/figs/e2_regret.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    w = csv.writer(f); w.writerow(["src/tgt"]+states)
    for src in states:
        row = [src]
        for tgt in states:
            vals = runs.get((src,tgt))
            if vals and tgt in own_best:
                row.append(f"{statistics.median(vals)/own_best[tgt]:.3f}")
            else:
                row.append("")
        w.writerow(row)
print("wrote", out)
EOF
python3 tools/build_regret.py
```

Then plot the matrix from `e2_regret.csv` (analyzer's `plot_regret` accepts a `pandas.DataFrame.read_csv`-compatible input — adapt or write a 20-line plotter).

For E4, write a per-step-CSV plotter (5–10 lines of pandas/matplotlib): one bar chart of total wall time per `(variant, rep)`, one line chart of `wall_dt_ms` vs `step`, faceted by variant, with `mean_speed` overlaid as a secondary axis to show the regime drift.

For E5, plot mean total wall time vs `tuning_budget` from each `e5_tb-*` CSV.

## 10 · Output verification

Before declaring done, confirm:

```bash
ls measurements/snapshots/*.sphs        | wc -l   # 7
ls measurements/configs/*.json          | wc -l   # 7
ls measurements/runs/e1_*.json          | wc -l   # 18  (6 states × 3)
ls measurements/runs/e2_*.json          | wc -l   # 126 (7 src × 6 tgt × 3)
ls measurements/runs/e4_*.json          | wc -l   # 20  (4 variants × 5)
ls measurements/runs/e4_*.steps.csv     | wc -l   # 20
ls measurements/runs/e5_*.steps.csv     | wc -l   # 18  (6 budgets × 3)
ls measurements/figs/                   # PNGs + CSVs
```

If any count is short, identify the failing run from stderr logs and rerun only that command.

## 11 · Failure modes to watch

- **Snapshot load mismatch.** `Snapshot::load` validates `app_name` and `fluid_n`. If you accidentally try `--snapshot-load fountain_active.sphs` with the `dragon_collision` binary, it will refuse — that's correct, not a bug.
- **Frozen config validation.** If a parameter name in a config JSON doesn't match a registered KTT parameter for that scenario, the run aborts. Cross-scenario configs are fine because every scenario uses the same `StepTuner` parameter set; `EXTERNAL_FORCE` is not in the tunable parameter set so it never appears in the JSONs.
- **Thermal drift.** If runs at the *end* of E2 or E4 are systematically slower than at the start, increase the `sleep` between runs. Inspect by plotting median per-replicate duration vs run order.
- **Searched count <200 in E1.** KTT may saturate the configuration space below 200 — that's expected; use whatever was produced.

## 12 · Total time budget

| phase                | runs | per-run | total       |
|----------------------|------|---------|-------------|
| Snapshots            | 7    | ~30 s   | ~4 min      |
| E1 (per-state MCMC)  | 18   | ~80 s   | ~25 min     |
| E2 (regret matrix)   | 126  | ~25 s   | ~55 min     |
| E4 (transient)       | 20   | ~75 s   | ~25 min     |
| E5 (budget sweep)    | 18   | ~75 s   | ~25 min     |
| Analysis & plots     | —    | —       | ~5 min      |
| **Total**            |      |         | **≈2.5 h**  |

Plus the cooldown sleeps (≈45 min total). Run overnight; one walltime budget of 4 h covers everything.

## 13 · Deliverables

When complete, the agent should leave:

- `measurements/snapshots/*.sphs` — reusable for future runs
- `measurements/configs/*.json` — per-state best configs
- `measurements/runs/*.json` — raw KTT result files
- `measurements/runs/*.steps.csv` — per-step logs for E4/E5
- `measurements/figs/*.png` and `*.csv` — analysis output
- `measurements/RUN_LOG.md` — append-only log of which commands ran when, with stderr summaries; create with `set -x; … 2>&1 | tee -a measurements/RUN_LOG.md`

Stop. Report counts from §10 and any failures.
