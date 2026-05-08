#include "tuner.h"
#include <memory>
#include <debug.h>
#include <cuda.h>

#include "config.h"


Tuner::Tuner(std::string name) : tuner{instance()}, name{std::move(name)} {}

Tuner::~Tuner() {
    if (searched_count > 0) {
        print_best_config(std::cout);
        tuner->SaveResults(results, cfg::results_dir / name, ktt::OutputFormat::JSON);
    }
}

std::pair<int, int> Tuner::tuning_stats() const {
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

void Tuner::update_args(const std::vector<ktt::ArgumentId>& new_args) {
    tuner->SetArguments(definition, new_args);
    for (const auto& arg : args)
        tuner->RemoveArgument(arg);
    args = new_args;
}

void Tuner::print_best_config(std::ostream& out) const {
    assert(tuner != nullptr);

    auto bestConfig = tuner->GetBestConfiguration(kernel);
    out << "Best configuration for " << name << ":\n";

    for (const auto& param : bestConfig.GetPairs()) {
        if (param.GetName() == "EXTERNAL_FORCE" || param.GetName() == "KERNEL_DIR")
            continue;

        out << "  " << param.GetName() << " = ";
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
