#pragma once

#include <cassert>
#include "config.h"
#include "tuner.h"
#include "cuda/nsearch/nsearch.cuh"


class DensityTuner final : public Tuner {
public:
    explicit DensityTuner(unsigned fluid_particles, unsigned total_particles) {
        assert(tuner != nullptr);

        const ktt::DimensionVector gridDimensions(std::bit_ceil(fluid_particles));
        const ktt::DimensionVector blockDimensions;

        static const std::string kernel_name = "compute_densities";
        definition = tuner->AddKernelDefinitionFromFile(kernel_name + (total_particles > fluid_particles ? "_with_boundary" : ""),
            cfg::tuned_kernels_dir / (kernel_name + ".cu"), gridDimensions, blockDimensions);
        kernel = tuner->CreateSimpleKernel(kernel_name, definition);

        tuner->SetSearcher(kernel, std::make_unique<ktt::McmcSearcher>());

        tuner->AddParameter<std::string>(kernel, "KERNEL_DIR", {cfg::tuned_kernels_dir});

        tuner->AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{32, 64, 128, 256, 512});
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Multiply);
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Divide);
    }

    ktt::KernelResult run(
        float4 *positions_dev_ptr, float* densities_dev_ptr,
        float* boundary_mass_dev_ptr, NSearch *fluid_n_search,
        unsigned total_particles, unsigned fluid_particles,
        NSearch *boundary_n_search, bool tune
    ) {
        std::vector args{
            tuner->AddArgumentVector<float4>(positions_dev_ptr, total_particles * sizeof(float4),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentVector<float>(densities_dev_ptr, fluid_particles * sizeof(float),
                ktt::ArgumentAccessType::WriteOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentScalar(fluid_particles),
            tuner->AddArgumentVector<NSearch>(fluid_n_search, sizeof(NSearch),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device)
        };
        if (total_particles > fluid_particles) {
            args.push_back(tuner->AddArgumentVector<float>(boundary_mass_dev_ptr, (total_particles - fluid_particles) * sizeof(float),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device));
            args.push_back(tuner->AddArgumentVector<NSearch>(boundary_n_search, sizeof(NSearch),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device));
        }
        update_args(args);
        return Tuner::run(tune);
    }
};
