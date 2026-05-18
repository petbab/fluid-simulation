# Fluid Simulation

Real-time 3D Smoothed Particle Hydrodynamics (SPH) fluid simulator.
CUDA-accelerated solver with dynamic autotuning via [KTT](https://github.com/HiPerCoRe/KTT),
rendered through OpenGL with CUDA–GL interop.

GitHub: https://github.com/petbab/fluid-simulation

## Features

- Weakly Compressible SPH (WCSPH) solver with CFL-adaptive time stepping
- User-controlled dynamic autotuning: MCMC / random / full-space searchers, configurable budget
- Spatial hashing via a custom CUDA neighborhood search (linear-probed hash table)
- Boundary handling from arbitrary triangle meshes (one-time boundary-particle sampling)
- ImGui control panel with multiple visualization modes (e.g., density, pressure)
- Binary snapshot save/load (`.sphs`) of fluid state
- Headless mode for reproducible benchmarking, with per-step CSV logging

## Scenes

Each subdirectory of `app/` is a standalone scene compiled as its own executable.

To add a scene, create `app/<name>/` containing `application.{h,cu}`, `main.cu`, and a `CMakeLists.txt`
with `register_app(<name>)`. Top-level CMake picks it up automatically via `GLOB`.

## Requirements

- CUDA Toolkit (runtime + headers) — [link](https://developer.nvidia.com/cuda-toolkit)
- C++20 compiler, CMake (CUDA C++20 needs ≥ 3.25 or the in-tree workaround)
- OpenGL 4.5
- vcpkg — provides `glad`, `glfw3`, `glm`, `imgui` (with glfw + opengl3 bindings), `nlohmann-json`, `gtest`. [setup](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started?pivots=shell-bash)
- [Open3D](https://github.com/isl-org/Open3D/tree/main/examples/cmake/open3d-cmake-find-package) (installed and findable by CMake)
- KTT — vendored as a submodule, built separately
- Debian/Ubuntu system packages:
  ```
  sudo apt install libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev pkg-config
  ```

## Build

1. Initialize the KTT submodule:
   ```
   git submodule update --init --recursive
   ```
2. Build KTT — see [KTT/Readme.md#building-ktt](https://github.com/HiPerCoRe/KTT/blob/master/Readme.md#building-ktt).
3. Configure paths: copy `cmake/CMakeUserConfig.cmake.in` to `cmake/CMakeUserConfig.cmake` and set `VCPKG_ROOT`, `KTT_DIR`, etc.
4. Build:
   ```
   cmake -B build
   cmake --build build -j $(nproc)
   ```

Executables land in `build/bin/<scene>`. `results/` and `snapshots/` directories are created at configure time.

## Run

Interactive:
```
./build/bin/dragon_collision
```

Headless benchmark (no window/GUI, fixed dt, MCMC searcher, 10s of simulated time, per-step CSV):
```
./build/bin/dragon_collision \
    --headless --stop sim-time=10 \
    --searcher mcmc --tuning-budget 0.1 \
    --log-csv run.csv --log-metrics
```

## Controls (interactive mode)

| Input             | Action                            |
| ----------------- | --------------------------------- |
| Left mouse click  | Capture mouse (enter fly mode)    |
| Mouse             | Look                              |
| `W` `A` `S` `D`   | Move camera horizontally          |
| `Q` `E`           | Move camera down / up             |
| `Space`           | Pause / resume                    |
| `→` (Right arrow) | Step one frame while paused       |
| `R`               | Reset fluid                       |
| `B`               | Toggle boundary-particle display  |
| `Esc`             | Release mouse, or quit if released |

## CLI reference

```
--headless                       No window, no GUI, no render
--snapshot-load FILE             Load snapshot after setup_scene
--snapshot-save FILE             Save snapshot at stop
--frozen-config FILE             JSON config; disables searcher (run with fixed tuning params)
--searcher {mcmc|random|full}    Default: mcmc
--tuning-budget FLOAT            Fraction of steps spent tuning. Default: 0.1; 0 disables tuning
--stop iters=N | sim-time=T | wall-time=T   Required in --headless
--fixed-dt FLOAT                 Default: 0.01
--warmup-iters N                 Default: 0
--ktt-output PATH                Prefix for KTT SaveResults
--log-csv FILE                   Per-step CSV log
--log-metrics                    Append mean_speed,ke columns to CSV
```

## Layout

```
app/         scenes (one executable each)
src/
  cuda/SPH/         SPH solver, particle data, snapshots, visualizer
  cuda/nsearch/     GPU neighborhood search
  cuda/tuning/      KTT tuners, scheduler, tuned kernel sources
  render/           OpenGL renderer (camera, lights, geometry, shaders)
  simulation/       FluidSimulator base class
shaders/     GLSL shaders
models/      .obj assets (e.g. Stanford dragon)
snapshots/   .sphs snapshots
results/     KTT tuning output, benchmark CSVs
KTT/         submodule
```
