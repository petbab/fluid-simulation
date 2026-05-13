#pragma once

#include <cassert>
#include "config.h"
#include "tuner.h"
#include "cuda/nsearch/nsearch.cuh"


/**
 * @brief KTT tuner for the compute_boundary_mass kernel.
 *
 * Computes boundary particle masses using the SPH density summation
 * with a fixed block size parameter.
 */
class ComputeBoundaryMassTuner final : public Tuner {
public:
    /**
     * @brief Constructs the tuner and registers the kernel definition.
     * @param boundary_particles Number of boundary particles.
     */
    explicit ComputeBoundaryMassTuner(unsigned boundary_particles) {
        assert(tuner != nullptr);

        const ktt::DimensionVector gridDimensions(std::bit_ceil(boundary_particles));
        const ktt::DimensionVector blockDimensions;

        static const std::string kernel_name = "compute_boundary_mass";
        definition = tuner->AddKernelDefinitionFromFile(
            kernel_name, cfg::tuned_kernels_dir / (kernel_name + ".cu"), gridDimensions, blockDimensions);
        kernel = tuner->CreateSimpleKernel(kernel_name, definition);

        tuner->SetSearcher(kernel, std::make_unique<ktt::McmcSearcher>());

        tuner->AddParameter<std::string>(kernel, "KERNEL_DIR", {cfg::tuned_kernels_dir});
        tuner->AddParameter<std::uint64_t>(kernel, "TABLE_SIZE", {std::bit_ceil(boundary_particles) / 2});

        tuner->AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{128});
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Multiply);
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Divide);
    }

    ~ComputeBoundaryMassTuner() override {
        searched_count = 0;
        results.clear();
    }

    /**
     * @brief Runs the boundary mass computation kernel.
     * @param positions_dev_ptr Device pointer to boundary positions.
     * @param masses_dev_ptr Device pointer to output masses.
     * @param boundary_n Number of boundary particles.
     * @param boundary_n_search Device neighbor search structure.
     * @param tune If true, runs the KTT tuner.
     * @return Kernel result from KTT.
     */
    ktt::KernelResult run(
        float4* positions_dev_ptr, float* masses_dev_ptr, unsigned boundary_n,
        NSearch* boundary_n_search, bool tune
    ) {
        std::vector args{
            tuner->AddArgumentVector<float4>(positions_dev_ptr, boundary_n * sizeof(float4),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentVector<float>(masses_dev_ptr, boundary_n * sizeof(float),
                ktt::ArgumentAccessType::WriteOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentScalar(boundary_n),
            tuner->AddArgumentVector<NSearch>(boundary_n_search, sizeof(NSearch),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device)
        };
        update_args(args);
        return Tuner::run(tune);
    }
};
