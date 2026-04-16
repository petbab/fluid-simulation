#pragma once

#include <cassert>
#include "config.h"
#include "tuner.h"


class ComputePressureTuner final : public Tuner {
public:
    explicit ComputePressureTuner(unsigned fluid_particles, float* densities, float* pressures)
        : fluid_particles(fluid_particles) {
        assert(tuner != nullptr);

        const ktt::DimensionVector gridDimensions(std::bit_ceil(fluid_particles));
        const ktt::DimensionVector blockDimensions;

        static const std::string kernel_name = "compute_pressure";
        definition = tuner->AddKernelDefinitionFromFile(
            kernel_name, cfg::tuned_kernels_dir / (kernel_name + ".cu"), gridDimensions, blockDimensions);
        kernel = tuner->CreateSimpleKernel(kernel_name, definition);

        tuner->AddParameter<std::string>(kernel, "KERNEL_DIR", {cfg::tuned_kernels_dir});

        tuner->AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{32, 64, 128, 256, 512});
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Multiply);
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Divide);

        densities_id = tuner->AddArgumentVector<float>(densities, fluid_particles * sizeof(float),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device);
        pressures_id = tuner->AddArgumentVector<float>(pressures, fluid_particles * sizeof(float),
                ktt::ArgumentAccessType::WriteOnly, ktt::ArgumentMemoryLocation::Device);
        fluid_particles_id = tuner->AddArgumentScalar(fluid_particles);
    }

    ktt::KernelResult run(bool tune) {
        tuner->SetArguments(definition, {
            densities_id,
            pressures_id,
            fluid_particles_id,
        });

        return Tuner::run(tune);
    }

private:
    unsigned fluid_particles;
    ktt::ArgumentId densities_id;
    ktt::ArgumentId pressures_id;
    ktt::ArgumentId fluid_particles_id;
};
