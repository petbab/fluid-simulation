#pragma once

#include <cassert>
#include "config.h"
#include "tuner.h"
#include "cuda/nsearch/nsearch.cuh"


class ComputeViscosityTuner final : public Tuner {
public:
    explicit ComputeViscosityTuner(unsigned fluid_particles) {
        assert(tuner != nullptr);

        const ktt::DimensionVector gridDimensions(std::bit_ceil(fluid_particles));
        const ktt::DimensionVector blockDimensions;

        static const std::string kernel_name = "compute_viscosity";
        definition = tuner->AddKernelDefinitionFromFile(
            kernel_name, cfg::tuned_kernels_dir / (kernel_name + ".cu"), gridDimensions, blockDimensions);
        kernel = tuner->CreateSimpleKernel(kernel_name, definition);

        tuner->AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{16, 32, 64, 128, 256, 512});
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Multiply);
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Divide);
    }

    void run(
        float* positions, float4* velocities,
        float* densities, float4* acceleration,
        unsigned fluid_particles, NSearch *dev_n_search
    ) {
        tuner->SetArguments(definition, {
            tuner->AddArgumentVector<float>(positions, fluid_particles * sizeof(float) * 3,
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentVector<float4>(velocities, fluid_particles * sizeof(float4),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentVector<float>(densities, fluid_particles * sizeof(float),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentVector<float4>(acceleration, fluid_particles * sizeof(float4),
                ktt::ArgumentAccessType::WriteOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentScalar(fluid_particles),
            tuner->AddArgumentVector<NSearch>(dev_n_search, sizeof(NSearch),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device)
        });

        ktt::KernelResult result = tuner->TuneIteration(kernel, {});
    }
};
