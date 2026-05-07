#pragma once
#include "config.h"
#include "tuner.h"
#include "cuda/nsearch/nsearch.cuh"


class StepTuner final : public Tuner {
public:
    StepTuner(unsigned fluid_particles, unsigned boundary_particles,
              std::string external_force = {})
        : fluid_n(fluid_particles),
        boundary_n(boundary_particles),
        total_n(fluid_particles + boundary_particles),
        has_boundary(boundary_particles > 0)
    {
        // populated via thread modifier
        const ktt::DimensionVector grid_size(fluid_n);
        const ktt::DimensionVector block_size;

        // rebuild_n_search may already be registered by NSearchWrapper.
        // KTT requires unique definition names, so reuse if present.
        def_rebuild_n_search = tuner->GetKernelDefinitionId("rebuild_n_search");
        if (def_rebuild_n_search == ktt::InvalidKernelDefinitionId) {
            def_rebuild_n_search = tuner->AddKernelDefinitionFromFile(
                "rebuild_n_search", cfg::tuned_kernels_dir / "rebuild_n_search.cu",
                grid_size, block_size);
        }

        const std::string density_name = has_boundary
            ? "compute_rho_p_with_boundary" : "compute_rho_p";
        def_compute_rho_p = tuner->AddKernelDefinitionFromFile(
            density_name, cfg::tuned_kernels_dir / "compute_rho_p.cu",
            grid_size, block_size);

        const std::string pforce_name = has_boundary
            ? "apply_pressure_force_n_normal_with_boundary" : "apply_pressure_force_n_normal";
        def_apply_pressure_force_n_normal = tuner->AddKernelDefinitionFromFile(
            pforce_name, cfg::tuned_kernels_dir / "apply_pressure_force_n_normal.cu",
            grid_size, block_size);

        def_compute_non_pressure_accel = tuner->AddKernelDefinitionFromFile(
            "compute_non_pressure_accel", cfg::tuned_kernels_dir / "compute_non_pressure_accel.cu",
            grid_size, block_size);

        std::vector defs{
            def_rebuild_n_search, def_compute_rho_p, def_apply_pressure_force_n_normal,
            def_compute_non_pressure_accel, //def_update_positions
        };
        kernel = tuner->CreateCompositeKernel(
            "simulation_step", defs,
            [this](ktt::ComputeInterface& iface) { launch(iface); });

        tuner->SetSearcher(kernel, std::make_unique<ktt::McmcSearcher>());

        // Shared compile-time params. Apply to all definitions in the composite.
        tuner->AddParameter<std::string>(kernel, "KERNEL_DIR", {cfg::tuned_kernels_dir});
        if (!external_force.empty())
            tuner->AddParameter(kernel, "EXTERNAL_FORCE", std::vector{std::move(external_force)});

        // Per-definition block size: each tunable independently inside the joint space.
        add_block_param("block_rebuild_n_search", def_rebuild_n_search);
        add_block_param("block_compute_rho_p", def_compute_rho_p);
        add_block_param("block_apply_pressure_force_n_normal", def_apply_pressure_force_n_normal);
        add_block_param("block_compute_non_pressure_accel", def_compute_non_pressure_accel);
    }

    // Caller must invoke fluid_n_search.clear() before this.
    // pressure_delta is the result of adapt_time_step from the previous frame's velocities.
    // non_pressure_delta is the caller-supplied delta capped at NON_PRESSURE_MAX_TIME_STEP.
    ktt::KernelResult run(
        NSearch* fluid_n_search, NSearch* boundary_n_search,
        float4* positions, float4* velocities,
        float* densities, float* pressures,
        float4* non_pressure_accel, float4* normals,
        float* boundary_mass, float pressure_delta,
        bool tune)
    {
        auto old_args = std::move(owned_args);
        owned_args.clear();

        // Buffers: register once per run, bind per definition below.
        // Access types are the union over all kernels reading/writing the buffer
        // within this composite invocation; KTT only validates this lazily.
        const auto a_fluid_search = vec_arg<NSearch>(
            fluid_n_search, sizeof(NSearch), ktt::ArgumentAccessType::ReadWrite);
        const auto a_positions = vec_arg<float4>(
            positions, total_n * sizeof(float4), ktt::ArgumentAccessType::ReadWrite);
        const auto a_velocities = vec_arg<float4>(
            velocities, fluid_n * sizeof(float4), ktt::ArgumentAccessType::ReadWrite);
        const auto a_densities = vec_arg<float>(
            densities, fluid_n * sizeof(float), ktt::ArgumentAccessType::ReadWrite);
        const auto a_pressures = vec_arg<float>(
            pressures, fluid_n * sizeof(float), ktt::ArgumentAccessType::ReadWrite);
        const auto a_np_accel = vec_arg<float4>(
            non_pressure_accel, fluid_n * sizeof(float4), ktt::ArgumentAccessType::ReadWrite);
        const auto a_normals = vec_arg<float4>(
            normals, fluid_n * sizeof(float4), ktt::ArgumentAccessType::ReadWrite);

        const auto a_n_fluid   = scalar_arg(fluid_n);
        const auto a_p_delta   = scalar_arg(pressure_delta);

        ktt::ArgumentId a_boundary_mass{}, a_boundary_search{};
        if (has_boundary) {
            a_boundary_mass = vec_arg<float>(
                boundary_mass, boundary_n * sizeof(float), ktt::ArgumentAccessType::ReadOnly);
            a_boundary_search = vec_arg<NSearch>(
                boundary_n_search, sizeof(NSearch), ktt::ArgumentAccessType::ReadOnly);
        }

        // Per-definition arg lists. Order must match each kernel's parameter list.
        tuner->SetArguments(def_rebuild_n_search, {a_fluid_search, a_positions, a_n_fluid});

        if (has_boundary)
            tuner->SetArguments(def_compute_rho_p,
                {a_positions, a_densities, a_pressures, a_n_fluid, a_fluid_search,
                 a_boundary_mass, a_boundary_search});
        else
            tuner->SetArguments(def_compute_rho_p,
                {a_positions, a_densities, a_pressures, a_n_fluid, a_fluid_search});

        if (has_boundary)
            tuner->SetArguments(def_apply_pressure_force_n_normal,
                {a_positions, a_densities, a_pressures, a_velocities, a_normals,
                 a_n_fluid, a_p_delta, a_fluid_search,
                 a_boundary_mass, a_boundary_search});
        else
            tuner->SetArguments(def_apply_pressure_force_n_normal,
                {a_positions, a_densities, a_pressures, a_velocities, a_normals,
                 a_n_fluid, a_p_delta, a_fluid_search});

        tuner->SetArguments(def_compute_non_pressure_accel,
            {a_positions, a_densities, a_velocities, a_normals,
                a_np_accel, a_n_fluid, a_fluid_search});

        ktt::KernelResult result = Tuner::run(tune);

        // Args are owned per-call; release them so the next invocation can rebind freshly.
        for (const auto& a : old_args) tuner->RemoveArgument(a);

        return result;
    }

private:
    void launch(ktt::ComputeInterface& iface) const {
        iface.RunKernel(def_rebuild_n_search);
        iface.RunKernel(def_compute_rho_p);
        iface.RunKernel(def_apply_pressure_force_n_normal);
        iface.RunKernel(def_compute_non_pressure_accel);
    }
    
    void add_block_param(const std::string& name, ktt::KernelDefinitionId def) const {
        // Each block-size parameter is its own independent group. Kernels run
        // sequentially inside the composite launcher, so one kernel's optimal
        // block size is independent of the others'.
        tuner->AddParameter(kernel, name, std::vector<uint64_t>{32, 64, 128, 256, 512}, name);
        tuner->AddThreadModifier(kernel, {def}, ktt::ModifierType::Local,
            ktt::ModifierDimension::X, name, ktt::ModifierAction::Multiply);
        tuner->AddThreadModifier(kernel, {def}, ktt::ModifierType::Global,
            ktt::ModifierDimension::X, name, ktt::ModifierAction::DivideCeil);
    }
    
    template <typename T>
    ktt::ArgumentId vec_arg(T* ptr, size_t bytes, ktt::ArgumentAccessType access) {
        auto id = tuner->AddArgumentVector<T>(
            ptr, bytes, access, ktt::ArgumentMemoryLocation::Device);
        owned_args.push_back(id);
        return id;
    }

    template <typename T>
    ktt::ArgumentId scalar_arg(T value) {
        auto id = tuner->AddArgumentScalar(value);
        owned_args.push_back(id);
        return id;
    }
    
    unsigned fluid_n, boundary_n, total_n;
    bool has_boundary;

    ktt::KernelDefinitionId def_rebuild_n_search = 0,
                            def_compute_rho_p = 0,
                            def_apply_pressure_force_n_normal = 0,
                            def_compute_non_pressure_accel = 0;

    std::vector<ktt::ArgumentId> owned_args;
};
