#pragma once

#include "tuner.h"
#include <render/box.h>
#include "config.h"
#include "cuda/util.cuh"


class UpdatePositionsTuner final : public Tuner {
public:
    explicit UpdatePositionsTuner(unsigned particles, float4* velocities_dev_ptr)
        : particles(particles) {
        assert(tuner != nullptr);

        const ktt::DimensionVector gridDimensions(std::bit_ceil(particles));
        const ktt::DimensionVector blockDimensions;

        static const std::string kernel_name = "update_positions";
        definition = tuner->AddKernelDefinitionFromFile(
            kernel_name, cfg::tuned_kernels_dir / (kernel_name + ".cu"), gridDimensions, blockDimensions);
        kernel = tuner->CreateSimpleKernel(kernel_name, definition);

        tuner->AddParameter<std::string>(kernel, "KERNEL_DIR", {cfg::tuned_kernels_dir});

        tuner->AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{32, 64, 128, 256, 512});
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Multiply);
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Divide);

        velocities_id = tuner->AddArgumentVector<float4>(velocities_dev_ptr, particles * sizeof(float4),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device);
        particles_id = tuner->AddArgumentScalar(particles);
    }

    ktt::KernelResult run(float *positions_dev_ptr, float delta, const BoundingBox &bb, bool tune) {
        tuner->SetArguments(definition, {
            tuner->AddArgumentVector<float>(positions_dev_ptr, particles * sizeof(float) * 3,
                ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device),
            velocities_id,
            particles_id,
            tuner->AddArgumentScalar(delta),
            tuner->AddArgumentScalar<BoundingBoxGPU>(bb)
        });

        return Tuner::run(tune);
    }

private:
    unsigned particles;
    ktt::ArgumentId velocities_id;
    ktt::ArgumentId particles_id;
};
