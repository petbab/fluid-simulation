#pragma once

#include <filesystem>

/**
 * @brief Compile-time configuration paths.
 *
 * All paths are derived from ROOT_DIR, which is injected by CMake.
 */
namespace cfg {

/** @brief Root directory of the project (set by CMake). */
static const std::filesystem::path root_dir{ROOT_DIR};

/** @brief Directory containing GLSL shader sources. */
static const std::filesystem::path shaders_dir = root_dir / "shaders";

/** @brief Directory containing 3D model files. */
static const std::filesystem::path models_dir = root_dir / "models";

/** @brief Directory containing auto-tuned CUDA kernel sources. */
static const std::filesystem::path tuned_kernels_dir = root_dir / "src/cuda/tuning/kernels";

/** @brief Directory for simulation result outputs. */
static const std::filesystem::path results_dir = root_dir / "results";

/** @brief Directory for simulation snapshot files. */
static const std::filesystem::path snapshots_dir = root_dir / "snapshots";

}
