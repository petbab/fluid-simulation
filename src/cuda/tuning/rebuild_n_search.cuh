#pragma once

#include "tuner.h"
#include "config.h"
#include "cuda/nsearch/nsearch.cuh"


class RebuildNSearchTuner final : public Tuner {
public:
    explicit RebuildNSearchTuner(unsigned total_particles) {
        assert(tuner != nullptr);

        const ktt::DimensionVector gridDimensions(std::bit_ceil(total_particles));
        const ktt::DimensionVector blockDimensions;

        static const std::string kernel_name = "rebuild_n_search";
        definition = tuner->AddKernelDefinitionFromFile(
            kernel_name, cfg::tuned_kernels_dir / (kernel_name + ".cu"), gridDimensions, blockDimensions);
        kernel = tuner->CreateSimpleKernel(kernel_name, definition);

        tuner->AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{16, 32, 64, 128, 256, 512});
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Multiply);
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Divide);
    }

    void run(NSearch *dev_n_search, float *particle_positions, unsigned total_particles) {
        tuner->SetArguments(definition, {
            tuner->AddArgumentVector<NSearch>(dev_n_search, sizeof(NSearch),
                ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentVector<float>(particle_positions, total_particles * sizeof(float) * 3,
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentScalar(total_particles),
        });

        ktt::KernelResult result = tuner->TuneIteration(kernel, {});
    }
};
