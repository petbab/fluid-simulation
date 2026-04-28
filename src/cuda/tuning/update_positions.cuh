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

        tuner->SetSearcher(kernel, std::make_unique<ktt::McmcSearcher>());

        tuner->AddParameter<std::string>(kernel, "KERNEL_DIR", {cfg::tuned_kernels_dir});

        tuner->AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{32, 64, 128, 256, 512});
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Multiply);
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Divide);
    }

    ktt::KernelResult run(float4 *positions_dev_ptr, float4* velocities_dev_ptr,
            unsigned n, float delta, const BoundingBox &bb, bool tune) {
        std::vector args{
            tuner->AddArgumentVector<float>(positions_dev_ptr, n * sizeof(float) * 3,
                ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentVector<float4>(velocities_dev_ptr, n * sizeof(float),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentScalar(n),
            tuner->AddArgumentScalar(delta),
            tuner->AddArgumentScalar<BoundingBoxGPU>(bb)
        };
        update_args(args);
        return Tuner::run(tune);
    }
};
