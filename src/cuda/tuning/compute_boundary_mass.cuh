#pragma once

#include <cassert>
#include "config.h"
#include "tuner.h"
#include "cuda/nsearch/nsearch.cuh"


class ComputeBoundaryMassTuner final : public Tuner {
public:
    explicit ComputeBoundaryMassTuner(unsigned boundary_particles) {
        assert(tuner != nullptr);

        const ktt::DimensionVector gridDimensions(std::bit_ceil(boundary_particles));
        const ktt::DimensionVector blockDimensions;

        static const std::string kernel_name = "compute_boundary_mass";
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
        float *positions_dev_ptr, float* masses_dev_ptr,
        unsigned fluid_n, unsigned boundary_n,
        NSearch *dev_n_search
    ) {
        tuner->SetArguments(definition, {
            tuner->AddArgumentVector<float>(positions_dev_ptr, (fluid_n + boundary_n) * sizeof(float) * 3,
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentVector<float>(masses_dev_ptr, boundary_n * sizeof(float),
                ktt::ArgumentAccessType::WriteOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentScalar(fluid_n),
            tuner->AddArgumentScalar(boundary_n),
            tuner->AddArgumentVector<NSearch>(dev_n_search, sizeof(NSearch),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device)
        });

        ktt::KernelResult result = tuner->TuneIteration(kernel, {});
    }
};
