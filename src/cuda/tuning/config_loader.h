#pragma once

#include <filesystem>
#include <Ktt.h>


ktt::KernelConfiguration load_config_json(const std::filesystem::path& path,
                                          ktt::Tuner& tuner,
                                          ktt::KernelId kernel);
