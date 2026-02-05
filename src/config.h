#pragma once

#include <filesystem>

namespace cfg {

static const std::filesystem::path root_dir{ROOT_DIR};

static const std::filesystem::path shaders_dir = root_dir / "shaders";

static const std::filesystem::path models_dir = root_dir / "models";

}
