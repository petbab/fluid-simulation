#include "density_tuner.cuh"

#include <config.h>
#include <bit>
#include <cassert>


static const std::string KERNEL_NAME = "compute_densities";
static const std::filesystem::path KERNEL_FILE = cfg::tuned_kernels_dir / (KERNEL_NAME + ".cu");

DensityTuner::DensityTuner(unsigned particles) {
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

struct NSearch{};

void DensityTuner::run(float *positions_dev_ptr, float* densities_dev_ptr,
        float* boundary_mass_dev_ptr, void *dev_n_search,
        unsigned total_particles, unsigned fluid_particles) {
    const ktt::ArgumentId positions_id = tuner->AddArgumentVector<float>(
        positions_dev_ptr, total_particles * sizeof(float) * 3,
        ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId densities_id = tuner->AddArgumentVector<float>(
        densities_dev_ptr, fluid_particles * sizeof(float),
        ktt::ArgumentAccessType::WriteOnly, ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId boundary_mass_id = tuner->AddArgumentVector<float>(
        boundary_mass_dev_ptr, (total_particles - fluid_particles) * sizeof(float),
        ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device);
    const ktt::ArgumentId n_id = tuner->AddArgumentScalar(fluid_particles);
    const ktt::ArgumentId dev_n_search_id = tuner->AddArgumentVector<NSearch>(
        dev_n_search, sizeof(NSearch),
        ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device);
    tuner->SetArguments(definition, {positions_id, densities_id, boundary_mass_id, n_id, dev_n_search_id});

    ktt::KernelResult result = tuner->TuneIteration(kernel, {});
}
