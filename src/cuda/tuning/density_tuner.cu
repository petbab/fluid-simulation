#include "density_tuner.cuh"

#include "../../debug.h"
#include "../../config.h"
#include <cuda.h>


static const std::string COMPUTE_DENSITIES_KERNEL = "compute_densities";

DensityTuner::DensityTuner(unsigned particles) {
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

    const ktt::DimensionVector blockDimensions{32};
    const ktt::DimensionVector gridDimensions{particles / blockDimensions.GetSizeX() + 1};

    definition = tuner->AddKernelDefinitionFromFile(
        COMPUTE_DENSITIES_KERNEL, cfg::tuned_kernels_dir / "density_kernel.cu",
        gridDimensions, blockDimensions
    );
    kernel = tuner->CreateSimpleKernel(COMPUTE_DENSITIES_KERNEL, definition);
}

void DensityTuner::run(float *positions_dev_ptr, float* densities_dev_ptr, unsigned particles) {
    // Add user-created buffer to tuner by providing its handle and size in bytes.
    const ktt::ArgumentId positions_id = tuner->AddArgumentVector<float>(
        positions_dev_ptr, particles * sizeof(float) * 3,
        ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId densities_id = tuner->AddArgumentVector<float>(
        densities_dev_ptr, particles * sizeof(float),
        ktt::ArgumentAccessType::WriteOnly, ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId n_id = tuner->AddArgumentScalar(particles);
    tuner->SetArguments(definition, {positions_id, densities_id, n_id});

    tuner->Run(kernel, {}, {});
}
