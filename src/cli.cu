#include "cli.h"
#include <iostream>
#include <cstring>
#include <cstdlib>


static void print_usage(const char* prog) {
    std::cerr <<
        "Usage: " << prog << " [OPTIONS]\n"
        "  --headless                      No window, no GUI, no render\n"
        "  --snapshot-load FILE            Load snapshot after setup_scene\n"
        "  --snapshot-save FILE            Save snapshot at stop\n"
        "  --frozen-config FILE            JSON config; disables searcher\n"
        "  --searcher {mcmc|random|full}   Default: mcmc\n"
        "  --tuning-budget FLOAT           Default: 0.1 (0 disables tuning)\n"
        "  --stop iters=N | sim-time=T | wall-time=T   Required in --headless\n"
        "  --fixed-dt FLOAT                Default: 0.01\n"
        "  --warmup-iters N                Default: 200\n"
        "  --ktt-output PATH               Prefix for tuner.SaveResults\n"
        "  --log-csv FILE                  Per-step CSV log\n"
        "  --log-metrics                   Add mean_speed,ke columns\n"
        "  --seed N                        Default: 42\n";
}

static RunOptions::Searcher parse_searcher(const char* s) {
    if (std::strcmp(s, "mcmc") == 0) return RunOptions::Searcher::Mcmc;
    if (std::strcmp(s, "random") == 0) return RunOptions::Searcher::Random;
    if (std::strcmp(s, "full") == 0) return RunOptions::Searcher::Full;
    throw std::invalid_argument(std::string("Unknown searcher: ") + s);
}

static void parse_stop(const char* s, RunOptions::StopKind& kind, double& value) {
    const char* eq = std::strchr(s, '=');
    if (!eq)
        throw std::invalid_argument("--stop must be iters=N, sim-time=T, or wall-time=T");

    std::string key(s, eq - s);
    const char* val = eq + 1;
    char* end = nullptr;
    value = std::strtod(val, &end);
    if (end == val || *end != '\0')
        throw std::invalid_argument(std::string("Invalid --stop value: ") + val);

    if (key == "iters") kind = RunOptions::StopKind::Iters;
    else if (key == "sim-time") kind = RunOptions::StopKind::SimTime;
    else if (key == "wall-time") kind = RunOptions::StopKind::WallTime;
    else throw std::invalid_argument(std::string("Unknown --stop kind: ") + key);
}

RunOptions parse_cli(int argc, char** argv) {
    RunOptions opts;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (std::strcmp(arg, "--headless") == 0) {
            opts.headless = true;
        } else if (std::strcmp(arg, "--snapshot-load") == 0) {
            if (++i >= argc) throw std::invalid_argument("--snapshot-load requires an argument");
            opts.snapshot_load = argv[i];
        } else if (std::strcmp(arg, "--snapshot-save") == 0) {
            if (++i >= argc) throw std::invalid_argument("--snapshot-save requires an argument");
            opts.snapshot_save = argv[i];
        } else if (std::strcmp(arg, "--frozen-config") == 0) {
            if (++i >= argc) throw std::invalid_argument("--frozen-config requires an argument");
            opts.frozen_config = argv[i];
        } else if (std::strcmp(arg, "--searcher") == 0) {
            if (++i >= argc) throw std::invalid_argument("--searcher requires an argument");
            opts.searcher = parse_searcher(argv[i]);
        } else if (std::strcmp(arg, "--tuning-budget") == 0) {
            if (++i >= argc) throw std::invalid_argument("--tuning-budget requires an argument");
            opts.tuning_budget = std::strtof(argv[i], nullptr);
        } else if (std::strcmp(arg, "--stop") == 0) {
            if (++i >= argc) throw std::invalid_argument("--stop requires an argument");
            parse_stop(argv[i], opts.stop_kind, opts.stop_value);
        } else if (std::strcmp(arg, "--fixed-dt") == 0) {
            if (++i >= argc) throw std::invalid_argument("--fixed-dt requires an argument");
            opts.fixed_dt = std::strtof(argv[i], nullptr);
        } else if (std::strcmp(arg, "--warmup-iters") == 0) {
            if (++i >= argc) throw std::invalid_argument("--warmup-iters requires an argument");
            opts.warmup_iters = std::atoi(argv[i]);
        } else if (std::strcmp(arg, "--ktt-output") == 0) {
            if (++i >= argc) throw std::invalid_argument("--ktt-output requires an argument");
            opts.ktt_output = argv[i];
        } else if (std::strcmp(arg, "--log-csv") == 0) {
            if (++i >= argc) throw std::invalid_argument("--log-csv requires an argument");
            opts.log_csv = argv[i];
        } else if (std::strcmp(arg, "--log-metrics") == 0) {
            opts.log_metrics = true;
        } else if (std::strcmp(arg, "--seed") == 0) {
            if (++i >= argc) throw std::invalid_argument("--seed requires an argument");
            opts.seed = std::strtoull(argv[i], nullptr, 10);
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument(std::string("Unknown argument: ") + arg);
        }
    }

    if (opts.headless && opts.stop_kind == RunOptions::StopKind::None)
        throw std::invalid_argument("--stop is required in --headless mode");

    return opts;
}
