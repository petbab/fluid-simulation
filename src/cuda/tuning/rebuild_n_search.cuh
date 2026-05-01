#pragma once

#include "tuner.h"
#include "config.h"
#include "cuda/nsearch/nsearch.cuh"


class RebuildNSearchTuner final : public Tuner {
public:
    explicit RebuildNSearchTuner(unsigned total_particles, bool boundary = false) {
        assert(tuner != nullptr);

        static const std::string kernel_name = "rebuild_n_search";

        if (boundary) {
            definition = tuner->GetKernelDefinitionId(kernel_name);
            kernel = tuner->CreateSimpleKernel(kernel_name + "_boundary", definition);
            auto def_id = definition;
            tuner->SetLauncher(kernel, [=](ktt::ComputeInterface& interface) {
                const ktt::DimensionVector gridDim{(total_particles + 127) / 128};
                const ktt::DimensionVector blockDim{128};
                interface.RunKernel(def_id, gridDim, blockDim);
            });
        } else {
            const ktt::DimensionVector gridDimensions(std::bit_ceil(total_particles));
            const ktt::DimensionVector blockDimensions;
            definition = tuner->AddKernelDefinitionFromFile(
                kernel_name, cfg::tuned_kernels_dir / (kernel_name + ".cu"),
                gridDimensions, blockDimensions);
            kernel = tuner->CreateSimpleKernel(kernel_name, definition);
            tuner->AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{32, 64, 128, 256, 512});
            tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Multiply);
            tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
                ktt::ModifierAction::Divide);
        }

        tuner->SetSearcher(kernel, std::make_unique<ktt::McmcSearcher>());

        tuner->AddParameter<std::string>(kernel, "KERNEL_DIR", {cfg::tuned_kernels_dir});
    }

    ktt::KernelResult run(NSearch *dev_n_search, float4 *particle_positions, unsigned total_particles, bool tune) {
        std::vector args{
            tuner->AddArgumentVector<NSearch>(dev_n_search, sizeof(NSearch),
                ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentVector<float4>(particle_positions, total_particles * sizeof(float4),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentScalar(total_particles),
        };
        update_args(args);
        return Tuner::run(tune);
    }
};
