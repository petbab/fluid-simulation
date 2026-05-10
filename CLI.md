# Minimal implementation plan for measurement infrastructure

## Scope

Add the bare-minimum hooks so each existing app binary (`devel`, `dragon_collision`, `fountain`, `surface_tension`, `tilting_box`, `vortex`) can be driven non-interactively. Reuse the existing `Snapshot` and `StepTuner`. No new tests, no benchmark framework.

## CLI surface (single shared parser, all apps)

```
<app_binary>
  --headless                      # no window, no GUI, no render
  --snapshot-load FILE            # passed to Snapshot::load after setup_scene
  --snapshot-save FILE            # taken at --stop
  --frozen-config FILE            # JSON config; disables searcher
  --searcher {mcmc|random|full}   # default mcmc
  --tuning-budget FLOAT           # default 0.1; 0 disables tuning
  --stop iters=N | sim-time=T | wall-time=T   # required in --headless
  --fixed-dt FLOAT                # default 0.01; ignored in interactive
  --warmup-iters N                # default 200; not measured
  --ktt-output PATH               # prefix for tuner.SaveResults; default keeps cfg::results_dir auto-naming
  --log-csv FILE                  # per-step CSV; default off
  --log-metrics                   # add mean_speed,ke columns (D->H reduce per step)
  --seed N                        # default 42
```

Exit code: 0 on stop reached, 1 on validation error, 2 on snapshot/config error.

## Yes — configuration selection is needed

For E2 (cross-state regret) and E4 (B0/B1/B2 baselines) you must run **arbitrary, externally-specified** configurations, not just KTT's current best. KTT supports this via `tuner->Run(kernel, KernelConfiguration, {})`; we just need a way to construct a `KernelConfiguration` from a file.

**Format** — same JSON schema KTT writes in result files, so configs can be lifted directly from a previous run's `Configuration` array:

```json
[
  {"Name": "CELL_SIZE_MULT",                       "Value": 0.6666, "ValueType": "Double"},
  {"Name": "TABLE_SIZE",                           "Value": 524288, "ValueType": "UnsignedInt"},
  {"Name": "BOUNDARY_TABLE_SIZE",                  "Value": 131072, "ValueType": "UnsignedInt"},
  {"Name": "compute_pressure_accel_n_normal_block","Value": 128,    "ValueType": "UnsignedInt"},
  {"Name": "compute_pressure_accel_n_normal_u_n",  "Value": 2,      "ValueType": "UnsignedInt"},
  {"Name": "compute_non_pressure_accel_block",     "Value": 32,     "ValueType": "UnsignedInt"},
  {"Name": "compute_non_pressure_accel_u_n",       "Value": 1,      "ValueType": "UnsignedInt"},
  {"Name": "compute_rho_p_block",                  "Value": 32,     "ValueType": "UnsignedInt"},
  {"Name": "compute_rho_p_u_n",                    "Value": 1,      "ValueType": "UnsignedInt"},
  {"Name": "count_neighbors_block",                "Value": 32,     "ValueType": "UnsignedInt"},
  {"Name": "count_neighbors_u_n",                  "Value": 1,      "ValueType": "UnsignedInt"},
  {"Name": "fill_neighbors_block",                 "Value": 32,     "ValueType": "UnsignedInt"},
  {"Name": "fill_neighbors_u_n",                   "Value": 1,      "ValueType": "UnsignedInt"},
  {"Name": "rebuild_n_search_block",               "Value": 32,     "ValueType": "UnsignedInt"},
  {"Name": "NEIGHBOR_LIST",                        "Value": 0,      "ValueType": "UnsignedInt"}
]
```

`KERNEL_DIR` and `EXTERNAL_FORCE` are filled in automatically (already constant per scenario). Missing parameters fall back to KTT's first-listed value via `Kernel::CreateConfiguration` — so `default.json` for B0 can simply be `[]`.

A small helper script `tools/extract_config.py results/result_07.json -> default.json` to dump the best-result config is one-page Python and worth including.

## File-by-file changes

### `src/cli.h`, `src/cli.cu`  (NEW)

```cpp
struct RunOptions {
    bool headless = false;
    std::optional<std::filesystem::path> snapshot_load;
    std::optional<std::filesystem::path> snapshot_save;
    std::optional<std::filesystem::path> frozen_config;
    enum class Searcher { Mcmc, Random, Full } searcher = Searcher::Mcmc;
    float tuning_budget = 0.1f;

    enum class StopKind { None, Iters, SimTime, WallTime } stop_kind = StopKind::None;
    double stop_value = 0.0;

    float fixed_dt = 0.01f;
    int warmup_iters = 200;

    std::optional<std::filesystem::path> ktt_output;
    std::optional<std::filesystem::path> log_csv;
    bool log_metrics = false;

    uint64_t seed = 42;
};

RunOptions parse_cli(int argc, char** argv);   // throw std::invalid_argument on bad input
```

Use `getopt_long` or hand-rolled — no new dependency. ~150 LOC.

### `app/*/main.cu` (each app, ~5-line edit)

Add CLI parse, set GLFW invisible window when headless, pass options through:

```cpp
int main(int argc, char** argv) {
    RunOptions opts = parse_cli(argc, argv);

    GLFW glfw{};
    if (opts.headless)
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);   // GL context still needed for CUDA-GL interop
    Window window{800, 600, APP_NAME};
    cuda_init();
    {
        DragonCollisionApp app{window.get(), window.width(), window.height(), APP_NAME, opts};
        app.init();
        app.run();
        AssetManager::free();
    }
    return EXIT_SUCCESS;
}
```

This pattern is identical across all 6 main.cu files. Could optionally factor out into a `RUN_APP(AppClass)` macro in a shared header.

### `src/application.h`, `src/application.cu`

- Constructor takes `RunOptions opts`; store as protected member.
- `init()` after `setup_scene()`:
  - if `opts.snapshot_load`: call `Snapshot::load` on the fluid sim's particle data.
  - if `opts.frozen_config`: `fluid_sim.set_frozen_config(load_config_json(...))`.
  - apply `opts.tuning_budget`, `opts.searcher` (the latter via a new `step_tuner.set_searcher(...)` — see below).
  - run `opts.warmup_iters` steps of `update_objects(opts.fixed_dt)`, **before** opening the CSV log and before clearing tuner results.
- New `void run_headless()`. Loop without GLFW poll/render/GUI:

```cpp
void Application::run_headless() {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    double sim_time = 0.0;
    int step = 0;

    std::ofstream log;
    if (opts.log_csv) {
        log.open(*opts.log_csv);
        log << "step,sim_time,wall_dt_ms,scheduled_tune";
        if (opts.log_metrics) log << ",mean_speed,ke";
        log << '\n';
    }

    while (!stop_reached(step, sim_time, t0)) {
        auto t_step = clock::now();
        update_objects(opts.fixed_dt);
        sim_time += opts.fixed_dt;
        auto wall_dt = std::chrono::duration<double, std::milli>(clock::now() - t_step).count();

        if (log) {
            auto* fluid_sim = ...;
            log << step << ',' << sim_time << ',' << wall_dt
                << ',' << (fluid_sim->was_scheduled_step() ? 1 : 0);
            if (opts.log_metrics) {
                auto [v, ke] = fluid_sim->compute_state_metrics();
                log << ',' << v << ',' << ke;
            }
            log << '\n';
        }
        ++step;
    }

    if (opts.snapshot_save)
        Snapshot::save(*opts.snapshot_save, app_name, fluid_sim->particle_data, fluid_sim->fluid_particles);
}
```

`run()` dispatches to `run_headless()` or the existing GLFW loop based on `opts.headless`.

`stop_reached()` is one switch on `opts.stop_kind`.

### `src/cuda/SPH/sph.cuh`, `src/cuda/SPH/sph.cu`

Three small additions:

1. `void set_frozen_config(ktt::KernelConfiguration cfg)` — stores it; `update()` then passes a "frozen" flag down to `step_tuner.run`.
2. `bool was_scheduled_step() const { return is_scheduled(STEP_TUNER); }` — for the CSV log.
3. `std::pair<float,float> compute_state_metrics() const` — reuses the existing `thrust::transform_reduce<float4_length_sq>` already in `adapt_time_step` for max-velocity; add one for mean-velocity and one for kinetic energy. Only called when `--log-metrics` is on.

### `src/cuda/tuning/tuner.h`, `src/cuda/tuning/tuner.cpp`

Extend the base `Tuner`:

```cpp
void set_searcher(RunOptions::Searcher s);     // calls tuner->SetSearcher(kernel, ...)
void set_frozen_config(ktt::KernelConfiguration cfg);
void clear_frozen_config();
```

Modify `Tuner::run(bool tune)`:

```cpp
ktt::KernelResult Tuner::run(bool tune) {
    if (frozen_config) {
        auto res = tuner->Run(kernel, *frozen_config, {});
        results.push_back(res);
        return res;
    }
    if (tune || searched_count == 0) {
        ++searched_count;
        return tuner->TuneIteration(kernel, {});
    }
    return tuner->Run(kernel, tuner->GetBestConfiguration(kernel), {});
}
```

Note: `set_searcher` must be called **before** any `TuneIteration`, and re-instantiates the searcher object — fine for our flow because we set it during `Application::init()` before warmup runs.

### `src/cuda/tuning/step_tuner.cuh`

Surface the new base methods:
- `set_searcher` is inherited.
- `set_frozen_config(...)` is inherited.

No other changes; the launcher already reads parameters out of `iface.GetCurrentConfiguration()` so a frozen config works the same way as a tuned one.

### `src/cuda/tuning/config_loader.h`, `.cu`  (NEW)

```cpp
ktt::KernelConfiguration load_config_json(const std::filesystem::path&,
                                          ktt::Tuner& tuner,
                                          ktt::KernelId kernel);
```

Parse the JSON array (or `{Configuration: [...]}` wrapper if present), build `ktt::ParameterInput`, call `tuner.CreateConfiguration(kernel, input)`. ValueType strings: `Double`, `UnsignedInt`, `Int`, `Bool`, `String` — matches `ktt::ParameterValueType`.

KTT internally uses a JSON serializer; check `Source/Output/JsonSerializer.{h,cpp}` and either reuse what it depends on or pull in `nlohmann/json` (header-only, single include). nlohmann/json is the lightest path; ~30 LOC of parsing.

### `src/gui.cu`

No-op when `opts.headless`. Easiest: in `Application` ctor, only construct `gui` when not headless; gate every `gui->...` call. ~5 lines.

### `cmake/RegisterApp.cmake`

No change needed — the new `cli.{h,cu}` and `config_loader.{h,cu}` go into `fluid_simulation_lib` which all apps already link against.

## Per-step CSV format

```
step,sim_time,wall_dt_ms,scheduled_tune
0,0.010,3.21,1
1,0.020,2.87,0
...
```

With `--log-metrics`:

```
step,sim_time,wall_dt_ms,scheduled_tune,mean_speed,ke
```

This is what the analyzer's `<basename>.state.csv` sidecar consumes (timestamp join is replaced by `step` join — adjust analyzer accordingly, trivial).

## Determinism notes (no code changes, but document)

- `--seed` is wired into `srand` and any RNG you introduce; current scenarios are already deterministic from grid init, but pass it to `RandomSearcher` constructor anyway.
- GPU clock locking is operator-side: `nvidia-smi -lgc <base>,<boost>` and `nvidia-smi -pm 1` before runs.
- Each binary invocation is a fresh `ktt::Tuner`, so no cross-run state leaks.

## Out of scope (deferred / not needed for thesis)

- Refactoring per-app `main.cu` into a single benchmark binary — boilerplate is small.
- Non-step tuners (`UpdatePositionsTuner`, `ComputeBoundaryMassTuner`) — the thesis story is the composite `simulation_step`; the others are not on the critical path.
- KTT MCP/profiling counters.
- Profile-guided sidecar metrics beyond mean speed and KE.
- Snapshot-versioning beyond what's already there.

## Estimated effort

| Component                          | LOC |
|------------------------------------|-----|
| `cli.{h,cu}`                       | ~150 |
| `Application` headless run + warmup | ~80 |
| `Tuner`/`StepTuner` frozen + searcher | ~40 |
| `config_loader.{h,cu}` (+ nlohmann/json) | ~60 |
| `sph.{cu,cuh}` metric methods       | ~30 |
| `main.cu` × 6                      | ~30 |
| `gui.cu` headless gate             | ~5 |
| **Total**                          | **~400** |

One afternoon for the agent.
