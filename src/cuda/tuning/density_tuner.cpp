#include "density_tuner.h"

#include "../../config.h"
#include <bit>
#include <cassert>


static const std::string COMPUTE_DENSITIES_KERNEL = "compute_densities";

DensityTuner::DensityTuner(unsigned particles) {
    assert(tuner != nullptr);

    const ktt::DimensionVector gridDimensions(std::bit_ceil(particles));
    const ktt::DimensionVector blockDimensions;

    definition = tuner->AddKernelDefinitionFromFile(
        COMPUTE_DENSITIES_KERNEL, cfg::tuned_kernels_dir / "density_kernel.cu",
        gridDimensions, blockDimensions
    );
    kernel = tuner->CreateSimpleKernel(COMPUTE_DENSITIES_KERNEL, definition);

    tuner->AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{16, 32, 64, 128, 256});
    tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
        ktt::ModifierAction::Multiply);
    tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
        ktt::ModifierAction::Divide);

    tuner->AddParameter(kernel, "UNROLL_FACTOR", std::vector<uint64_t>{1, 2, 4, 8, 16, 32});
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

    ktt::KernelResult result = tuner->TuneIteration(kernel, {});
}
