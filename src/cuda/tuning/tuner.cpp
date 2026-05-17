#include "tuner.h"
#include <memory>
#include <debug.h>
#include <cuda.h>
#include <filesystem>
#include "config.h"
#include <Api/Searcher/McmcSearcher.h>
#include <Api/Searcher/RandomSearcher.h>
#include <Api/Searcher/DeterministicSearcher.h>


static std::string result_name() {
    unsigned count = 0;
    for (const auto &de : std::filesystem::directory_iterator{cfg::results_dir})
        ++count;
    return (count > 9 ? "result_" : "result_0") + std::to_string(count);
}

Tuner::Tuner() : tuner{instance()} {}

Tuner::~Tuner() {
    if (searched_count > 0) {
        print_best_config(std::cout);
        if (results_out)
            tuner->SaveResults(results, *results_out, ktt::OutputFormat::JSON);
        else
            tuner->SaveResults(results, cfg::results_dir / result_name(), ktt::OutputFormat::JSON);
    }
}

std::pair<int, int> Tuner::tuning_stats() const {
    if (frozen_config)
        return {1, 1};

    if (searched_count == 0)
        return {0, 0};

    int total = tuner->GetConfigurationsCount(kernel);
    return {std::min(searched_count, total), total};
}

void Tuner::clear_configuration_data() {
    tuner->ClearConfigurationData(kernel);
    searched_count = 0;
    results.clear();
}

ktt::KernelResult Tuner::run(bool tune) {
    if (frozen_config) {
        auto res = tuner->Run(kernel, *frozen_config, {});
        results.push_back(res);
        return res;
    }
    ktt::KernelResult res;
    if (tune || searched_count == 0) {
        ++searched_count;
        res = tuner->TuneIteration(kernel, {});
    } else {
        auto best_configuration = tuner->GetBestConfiguration(kernel);
        res = tuner->Run(kernel, best_configuration, {});
    }
    results.push_back(res);
    return res;
}

void Tuner::set_searcher(RunOptions::Searcher s) {
    std::unique_ptr<ktt::Searcher> searcher;
    switch (s) {
    case RunOptions::Searcher::Mcmc:
        searcher = std::make_unique<ktt::McmcSearcher>();
        break;
    case RunOptions::Searcher::Random:
        searcher = std::make_unique<ktt::RandomSearcher>();
        break;
    case RunOptions::Searcher::Full:
        searcher = std::make_unique<ktt::DeterministicSearcher>();
        break;
    }
    tuner->SetSearcher(kernel, std::move(searcher));
}

void Tuner::set_results_out(std::optional<std::filesystem::path> out) {
    results_out = out;
}

void Tuner::set_frozen_config(ktt::KernelConfiguration cfg) {
    searched_count = 1;
    frozen_config = std::move(cfg);
}

void Tuner::clear_frozen_config() {
    searched_count = 0;
    frozen_config.reset();
}

void Tuner::update_args(const std::vector<ktt::ArgumentId>& new_args) {
    tuner->SetArguments(definition, new_args);
    for (const auto& arg : args)
        tuner->RemoveArgument(arg);
    args = new_args;
}

void Tuner::print_best_config(std::ostream& out) const {
    assert(tuner != nullptr);

    auto bestConfig = frozen_config ? *frozen_config : tuner->GetBestConfiguration(kernel);
    // out << "Best configuration for " << name << ":\n";

    for (const auto& param : bestConfig.GetPairs()) {
        if (param.GetName() == "EXTERNAL_FORCE" || param.GetName() == "KERNEL_DIR")
            continue;

        out << param.GetName() << " = ";
        std::visit([&](auto&& arg){ out << arg; }, param.GetValue());
        out << '\n';
    }
}

ktt::Tuner* Tuner::instance() {
    static std::unique_ptr<ktt::Tuner> tuner;
    if (tuner == nullptr) {
        CUcontext context;
        cuCtxGetCurrent(&context);
        cudaCheckError();

        CUstream stream;
        cuStreamCreate(&stream, CU_STREAM_DEFAULT);
        cudaCheckError();

#ifdef DEBUG
        ktt::Tuner::SetLoggingLevel(ktt::LoggingLevel::Info);
#else
        ktt::Tuner::SetLoggingLevel(ktt::LoggingLevel::Warning);
#endif

        // Create compute API initializer which specifies context and streams that will be utilized by the tuner.
        ktt::ComputeApiInitializer initializer{context, std::vector<ktt::ComputeQueue>{stream}};
        tuner = std::make_unique<ktt::Tuner>(ktt::ComputeApi::CUDA, initializer);
        tuner->SetCompilerOptions("--std=c++20");
	    tuner->SetKernelCacheCapacity(20);
    }
    return tuner.get();
}
