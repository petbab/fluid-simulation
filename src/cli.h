#pragma once

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>


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

RunOptions parse_cli(int argc, char** argv);
