#pragma once

#include <cassert>
#include "config.h"
#include "tuner.h"
#include "cuda/nsearch/nsearch.cuh"


class ApplyPressureForceTuner final : public Tuner {
public:
    explicit ApplyPressureForceTuner(unsigned fluid_particles, unsigned boundary_particles) {
        assert(tuner != nullptr);

        const ktt::DimensionVector gridDimensions(std::bit_ceil(fluid_particles));
        const ktt::DimensionVector blockDimensions;

        static const std::string kernel_name = "apply_pressure_force";
        definition = tuner->AddKernelDefinitionFromFile(kernel_name + (boundary_particles > 0 ? "_with_boundary" : ""),
            cfg::tuned_kernels_dir / (kernel_name + ".cu"), gridDimensions, blockDimensions);
        kernel = tuner->CreateSimpleKernel(kernel_name, definition);

        tuner->SetSearcher(kernel, std::make_unique<ktt::McmcSearcher>());

        tuner->AddParameter<std::string>(kernel, "KERNEL_DIR", {cfg::tuned_kernels_dir});

        tuner->AddParameter(kernel, "multiply_block_size", std::vector<uint64_t>{32, 64, 128, 256, 512});
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Local, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Multiply);
        tuner->AddThreadModifier(kernel, {definition}, ktt::ModifierType::Global, ktt::ModifierDimension::X, "multiply_block_size",
            ktt::ModifierAction::Divide);
    }

    ktt::KernelResult run(
        float4* positions, float* densities,
        float* pressures, float4* velocities,
        float* boundary_mass, unsigned fluid_particles,
        unsigned boundary_particles, float delta, NSearch *fluid_n_search,
        NSearch *boundary_n_search, bool tune
    ) {
        std::vector args{
            tuner->AddArgumentVector<float4>(positions, fluid_particles * sizeof(float4),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentVector<float>(densities, fluid_particles * sizeof(float),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentVector<float>(pressures, fluid_particles * sizeof(float),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentVector<float4>(velocities, fluid_particles * sizeof(float4),
                ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device),
            tuner->AddArgumentScalar(fluid_particles),
            tuner->AddArgumentScalar(delta),
            tuner->AddArgumentVector<NSearch>(fluid_n_search, sizeof(NSearch),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device)
        };

        if (boundary_particles > 0) {
            args.push_back(tuner->AddArgumentVector<float>(boundary_mass, boundary_particles * sizeof(float),
                ktt::ArgumentAccessType::ReadWrite, ktt::ArgumentMemoryLocation::Device));
            args.push_back(tuner->AddArgumentVector<NSearch>(boundary_n_search, sizeof(NSearch),
                ktt::ArgumentAccessType::ReadOnly, ktt::ArgumentMemoryLocation::Device));
        }

        update_args(args);
        return Tuner::run(tune);
    }
};
