#pragma once

#include <cassert>
#include "tuner.h"
#include "config.h"


class UpdateVelocitiesTuner final : public Tuner {
public:
    explicit UpdateVelocitiesTuner(unsigned fluid_particles, float4* velocities, float4* acceleration)
        : fluid_particles(fluid_particles) {
        assert(tuner != nullptr);

        const ktt::DimensionVector gridDimensions(std::bit_ceil(fluid_particles));
        const ktt::DimensionVector blockDimensions;

        static const std::string kernel_name = "update_velocities";
        definition = tuner->AddKernelDefinitionFromFile(
            kernel_name, cfg::tuned_kernels_dir / (kernel_name + ".cu"), gridDimensions, blockDimensions);
        kernel = tuner->CreateSimpleKernel(kernel_name, definition);

        tuner->AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{32, 64, 128, 256, 512});
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Multiply);
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Divide);

        velocities_id = tuner->AddArgumentVector<float4>(velocities, fluid_particles * sizeof(float4),
                ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device);
        acceleration_id = tuner->AddArgumentVector<float4>(acceleration, fluid_particles * sizeof(float4),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device);
        fluid_particles_id = tuner->AddArgumentScalar(fluid_particles);
    }

    ktt::KernelResult run(float delta, bool tune) {
        tuner->SetArguments(definition, {
            velocities_id,
            acceleration_id,
            fluid_particles_id,
            tuner->AddArgumentScalar(delta),
        });

        return Tuner::run(tune);
    }

private:
    unsigned fluid_particles;
    ktt::ArgumentId velocities_id;
    ktt::ArgumentId acceleration_id;
    ktt::ArgumentId fluid_particles_id;
};
