#pragma once

#include <filesystem>

namespace cfg {

const std::filesystem::path root_dir{ROOT_DIR};

const std::filesystem::path shaders_dir = root_dir / "shaders";

const std::filesystem::path tuned_kernels_dir = root_dir / "src/cuda/tuning/kernels";

}
