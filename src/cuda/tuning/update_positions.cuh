#pragma once

#include "tuner.h"
#include <render/box.h>
#include "config.h"
#include "cuda/util.cuh"


class UpdatePositionsTuner final : public Tuner {
public:
    explicit UpdatePositionsTuner(unsigned particles) {
        assert(tuner != nullptr);

        const ktt::DimensionVector gridDimensions(std::bit_ceil(particles));
        const ktt::DimensionVector blockDimensions;

        static const std::string kernel_name = "update_positions";
        definition = tuner->AddKernelDefinitionFromFile(
            kernel_name, cfg::tuned_kernels_dir / (kernel_name + ".cu"), gridDimensions, blockDimensions);
        kernel = tuner->CreateSimpleKernel(kernel_name, definition);

        tuner->AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{16, 32, 64, 128, 256, 512});
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Multiply);
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Divide);
    }

    void run(float *positions_dev_ptr, float4* velocities_dev_ptr,
            unsigned n, float delta, const BoundingBox &bb) {
        tuner->SetArguments(definition, {
            tuner->AddArgumentVector<float>(positions_dev_ptr, n * sizeof(float) * 3,
                ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentVector<float4>(velocities_dev_ptr, n * sizeof(float),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentScalar(n),
            tuner->AddArgumentScalar(delta),
            tuner->AddArgumentScalar<BoundingBoxGPU>(bb)
        });

        ktt::KernelResult result = tuner->TuneIteration(kernel, {});
    }
};
