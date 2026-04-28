#pragma once

#include <cassert>
#include "config.h"
#include "tuner.h"
#include "cuda/nsearch/nsearch.cuh"


class ComputeSurfaceNormalsTuner final : public Tuner {
public:
    explicit ComputeSurfaceNormalsTuner(unsigned fluid_particles) {
        assert(tuner != nullptr);

        const ktt::DimensionVector gridDimensions(std::bit_ceil(fluid_particles));
        const ktt::DimensionVector blockDimensions;

        static const std::string kernel_name = "compute_surface_normals";
        definition = tuner->AddKernelDefinitionFromFile(
            kernel_name, cfg::tuned_kernels_dir / (kernel_name + ".cu"), gridDimensions, blockDimensions);
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
        float4* positions, float* densities,
        float4* normals, unsigned fluid_particles,
        NSearch *dev_n_search, bool tune
    ) {
        std::vector args{
            tuner->AddArgumentVector<float>(positions, fluid_particles * sizeof(float) * 3,
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentVector<float>(densities, fluid_particles * sizeof(float),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentVector<float4>(normals, fluid_particles * sizeof(float4),
                ktt::ArgumentAccessType::WriteOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentScalar(fluid_particles),
            tuner->AddArgumentVector<NSearch>(dev_n_search, sizeof(NSearch),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device)
        };
        update_args(args);
        return Tuner::run(tune);
    }
};
