#include "tuner.h"
#include <memory>
#include "../../debug.h"
#include <cuda.h>

Tuner::Tuner() : tuner{instance()} {}

Tuner::~Tuner() {
    print_best_config();
}

void Tuner::print_best_config() const {
    assert(tuner != nullptr);

    auto bestConfig = tuner->GetBestConfiguration(kernel);
    std::cout << "Best configuration:" << std::endl;

    for (const auto& param : bestConfig.GetPairs()) {
        std::cout << "  " << param.GetName() << " = "
            << std::get<uint64_t>(param.GetValue()) << std::endl;
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
    }
    return tuner.get();
}
