#pragma once

#include <cassert>
#include "config.h"
#include "tuner.h"
#include "cuda/nsearch/nsearch.cuh"


class ComputeBoundaryMassTuner final : public Tuner {
public:
    explicit ComputeBoundaryMassTuner(unsigned fluid_n, unsigned boundary_n,
        float* masses_dev_ptr, NSearch *dev_n_search)
        : fluid_n(fluid_n), boundary_n(boundary_n), total_n(fluid_n + boundary_n) {
        assert(tuner != nullptr);

        const ktt::DimensionVector gridDimensions(std::bit_ceil(boundary_n));
        const ktt::DimensionVector blockDimensions;

        static const std::string kernel_name = "compute_boundary_mass";
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

        masses_id = tuner->AddArgumentVector<float>(masses_dev_ptr, boundary_n * sizeof(float),
                ktt::ArgumentAccessType::WriteOnly, ktt::ArgumentMemoryLocation::Device);
        fluid_n_id = tuner->AddArgumentScalar(fluid_n);
        boundary_n_id = tuner->AddArgumentScalar(boundary_n);
        n_search_id = tuner->AddArgumentVector<NSearch>(dev_n_search, sizeof(NSearch),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device);
    }

    ktt::KernelResult run(float *positions_dev_ptr, bool tune) {
        tuner->SetArguments(definition, {
            tuner->AddArgumentVector<float>(positions_dev_ptr, total_n * sizeof(float) * 3,
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            masses_id,
            fluid_n_id,
            boundary_n_id,
            n_search_id
        });

        return Tuner::run(tune);
    }

private:
    unsigned fluid_n;
    unsigned boundary_n;
    unsigned total_n;
    ktt::ArgumentId masses_id;
    ktt::ArgumentId fluid_n_id;
    ktt::ArgumentId boundary_n_id;
    ktt::ArgumentId n_search_id;
};
