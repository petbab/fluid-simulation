#include "update_positions_tuner.cuh"

#include <config.h>
#include <bit>
#include <cassert>
#include <cuda/util.cuh>


static const std::string KERNEL_NAME = "update_positions";
static const std::filesystem::path KERNEL_FILE = cfg::tuned_kernels_dir / (KERNEL_NAME + ".cu");

UpdatePositionsTuner::UpdatePositionsTuner(unsigned particles) {
    assert(tuner != nullptr);

    const ktt::DimensionVector gridDimensions(std::bit_ceil(particles));
    const ktt::DimensionVector blockDimensions;

    definition = tuner->AddKernelDefinitionFromFile(
        KERNEL_NAME, KERNEL_FILE, gridDimensions, blockDimensions);
    kernel = tuner->CreateSimpleKernel(KERNEL_NAME, definition);

    tuner->AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{16, 32, 64, 128, 256, 512});
    tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
        ktt::ModifierAction::Multiply);
    tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
        ktt::ModifierAction::Divide);
}

void UpdatePositionsTuner::run(float *positions_dev_ptr, float4* velocities_dev_ptr,
        unsigned n, float delta, const BoundingBox &bb) {
    const ktt::ArgumentId positions_id = tuner->AddArgumentVector<float>(
        positions_dev_ptr, n * sizeof(float) * 3,
        ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId velocities_id = tuner->AddArgumentVector<float4>(
        velocities_dev_ptr, n * sizeof(float),
        ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId n_id = tuner->AddArgumentScalar(n);
    const ktt::ArgumentId delta_id = tuner->AddArgumentScalar(delta);
    const ktt::ArgumentId dev_n_search_id = tuner->AddArgumentScalar<BoundingBoxGPU>(bb);
    tuner->SetArguments(definition, {positions_id, velocities_id, n_id, delta_id, dev_n_search_id});

    ktt::KernelResult result = tuner->TuneIteration(kernel, {});
}
