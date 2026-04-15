#pragma once

#include <cassert>
#include "config.h"
#include "tuner.h"
#include "cuda/nsearch/nsearch.cuh"


class DensityTuner final : public Tuner {
public:
    explicit DensityTuner(unsigned fluid_particles, unsigned total_particles, float* densities_dev_ptr,
            float* boundary_mass_dev_ptr, NSearch *dev_n_search)
        : fluid_particles(fluid_particles), total_particles(total_particles) {
        assert(tuner != nullptr);

        const ktt::DimensionVector gridDimensions(std::bit_ceil(fluid_particles));
        const ktt::DimensionVector blockDimensions;

        static const std::string kernel_name = "compute_densities";
        definition = tuner->AddKernelDefinitionFromFile(kernel_name + (total_particles > fluid_particles ? "_with_boundary" : ""),
            cfg::tuned_kernels_dir / (kernel_name + ".cu"), gridDimensions, blockDimensions);
        kernel = tuner->CreateSimpleKernel(kernel_name, definition);

        tuner->AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{32, 64, 128, 256, 512});
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Multiply);
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Divide);

        densities_id = tuner->AddArgumentVector<float>(densities_dev_ptr, fluid_particles * sizeof(float),
                ktt::ArgumentAccessType::WriteOnly, ktt::ArgumentMemoryLocation::Device);
        fluid_particles_id = tuner->AddArgumentScalar(fluid_particles);
        n_search_id = tuner->AddArgumentVector<NSearch>(dev_n_search, sizeof(NSearch),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device);

        if (total_particles > fluid_particles)
            boundary_mass_id =
                tuner->AddArgumentVector<float>(boundary_mass_dev_ptr, total_particles - fluid_particles * sizeof(float),
                    ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device);
    }

    ktt::KernelResult run(float *positions_dev_ptr, bool tune) {
        std::vector args{
            tuner->AddArgumentVector<float>(positions_dev_ptr, total_particles * sizeof(float) * 3,
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            densities_id,
            fluid_particles_id,
            n_search_id
        };
        if (total_particles > fluid_particles)
            args.push_back(boundary_mass_id);

        tuner->SetArguments(definition, args);

        return Tuner::run(tune);
    }

private:
    unsigned fluid_particles;
    unsigned total_particles;
    ktt::ArgumentId densities_id;
    ktt::ArgumentId boundary_mass_id;
    ktt::ArgumentId fluid_particles_id;
    ktt::ArgumentId n_search_id;
};
