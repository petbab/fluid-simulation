#pragma once

#include <cassert>
#include "tuner.h"
#include "config.h"


class ApplyExternalForcesTuner final : public Tuner {
public:
    explicit ApplyExternalForcesTuner(unsigned particles, std::string external_force) {
        assert(tuner != nullptr);

        const ktt::DimensionVector gridDimensions(std::bit_ceil(particles));
        const ktt::DimensionVector blockDimensions;

        static const std::string kernel_name = "apply_external_forces";
        definition = tuner->AddKernelDefinitionFromFile(
            kernel_name, cfg::tuned_kernels_dir / (kernel_name + ".cu"), gridDimensions, blockDimensions);
        kernel = tuner->CreateSimpleKernel(kernel_name, definition);

        tuner->AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{16, 32, 64, 128, 256, 512});
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Multiply);
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Divide);

        if (!external_force.empty())
            tuner->AddParameter(kernel, "EXTERNAL_FORCE", std::vector{std::move(external_force)});
    }

    void run(float* positions, float4* acceleration, unsigned fluid_particles) {
        tuner->SetArguments(definition, {
            tuner->AddArgumentVector<float>(positions, fluid_particles * sizeof(float) * 3,
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentVector<float4>(acceleration, fluid_particles * sizeof(float4),
                ktt::ArgumentAccessType::WriteOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentScalar(fluid_particles),
        });

        ktt::KernelResult result = tuner->TuneIteration(kernel, {});
    }
};
