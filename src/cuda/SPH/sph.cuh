#pragma once

#include <thrust/device_vector.h>
#include <cuda/nsearch/nsearch.h>
#include <cuda/tuning/compute_densities.cuh>
#include <cuda/tuning/update_positions.cuh>
#include "cuda/tuning/apply_external_forces.cuh"
#include "cuda/tuning/apply_pressure_force.cuh"
#include "cuda/tuning/compute_boundary_mass.cuh"
#include "cuda/tuning/compute_pressure.cuh"
#include "cuda/tuning/compute_surface_normals.cuh"
#include "cuda/tuning/compute_surface_tension.cuh"
#include "cuda/tuning/compute_viscosity.cuh"
#include "cuda/tuning/tuning_scheduler.h"
#include "cuda/tuning/update_velocities.cuh"
#include <memory>
#include "particle_data.cuh"
#include "particle_data_visualizer.cuh"
#include "simulation/fluid_simulator.h"


class CUDASPHSimulator final : public FluidSimulator {
public:
    ///////////////////////////////////////////////////////////////////////////////
    ////                         SIMULATION PARAMETERS                         ////
    ///////////////////////////////////////////////////////////////////////////////
    static constexpr float REST_DENSITY = 1000.f;
    static constexpr float SUPPORT_RADIUS = 2.f * PARTICLE_SPACING;
    static constexpr float PARTICLE_VOLUME = PARTICLE_SPACING * PARTICLE_SPACING * PARTICLE_SPACING * 0.8;
    static constexpr float PARTICLE_MASS = REST_DENSITY * PARTICLE_VOLUME;

    static constexpr float XSPH_ALPHA = 0.f;

    static constexpr float4 GRAVITY{0, -9.81f, 0, 0};

    static constexpr float CFL_FACTOR = 0.4f;
    static constexpr float NON_PRESSURE_MAX_TIME_STEP = 0.015;

    static constexpr float MAX_TIME_STEP = 0.0005f;
    static constexpr float MIN_TIME_STEP = 0.00001f;
    ///////////////////////////////////////////////////////////////////////////////

    CUDASPHSimulator(const opts_t& opts);

    void init_positions(GLuint pos_vao_a, GLuint pos_vao_b);

    void update(float delta) override;

    void visualize(Shader* shader) override;

    void reset() override;

private:
    void init_boundary();
    void build_boundary_n_search(float4* positions_dev_ptr);
    bool has_boundary() const;

    void compute_densities(float4* positions_dev_ptr);
    void compute_boundary_mass(float4* positions_dev_ptr);

    void update_positions(float4* positions_dev_ptr, float delta);
    void update_velocities(float delta);

    // Adapt the time step size according to the Courant-Friedrich-Levy (CFL) condition
    float adapt_time_step(float delta, float min_step, float max_step) const;

    /**
     * Simulates viscosity by smoothing the velocity field [SPH Tutorial, eq. 103].
     */
    void compute_XSPH(const float4* positions_dev_ptr);

    /**
     * Computes and applies the viscous force using an explicit viscosity model.
     * Approximates the Laplacian of the velocity field via finite differences
     * [SPH Tutorial, eq. 102].
     */
    void compute_viscosity(float4* positions_dev_ptr);

    /**
     * Simulates surface tension using the macroscopic approach by Akinci et al. (2013).
     */
    void compute_surface_tension(float4* positions_dev_ptr);

    /**
     * Computes surface normals used to calculate the curvature force
     * in compute_surface_tension [SPH Tutorial, eq. 125].
     */
    void compute_surface_normals(float4* positions_dev_ptr);

    void apply_non_pressure_forces(float4* positions_dev_ptr, float delta);
    void apply_external_forces(float4* positions_dev_ptr);

    void compute_pressure();
    void apply_pressure_force(float4* positions_dev_ptr, float delta);

    enum tuners {
        DENSITY_TUNER,
        UPDATE_POSITIONS_TUNER,
        UPDATE_VELOCITIES_TUNER,
        COMPUTE_VISCOSITY_TUNER,
        COMPUTE_SURFACE_NORMALS_TUNER,
        COMPUTE_SURFACE_TENSION_TUNER,
        COMPUTE_PRESSURE_TUNER,
        APPLY_PRESSURE_FORCE_TUNER,
        APPLY_EXTERNAL_FORCES_TUNER,
        REBUILD_N_SEARCH_TUNER,
    };

    bool is_scheduled(tuners tuner_i) const;

    std::pair<int, int> tuning_stats() const;

    void set_tuning_budget(float tuning_budget);
    void reset_tuning();

    ParticleData particle_data;
    ParticleDataVisualizer particle_data_visualizer;

    NSearchWrapper fluid_n_search;
    std::unique_ptr<NSearchWrapper> boundary_n_search;

    DensityTuner density_tuner;
    UpdatePositionsTuner update_positions_tuner;
    UpdateVelocitiesTuner update_velocities_tuner;
    ComputeViscosityTuner compute_viscosity_tuner;
    ComputeSurfaceNormalsTuner compute_surface_normals_tuner;
    ComputeSurfaceTensionTuner compute_surface_tension_tuner;
    ComputePressureTuner compute_pressure_tuner;
    ApplyPressureForceTuner apply_pressure_force_tuner;

    std::unique_ptr<ComputeBoundaryMassTuner> compute_boundary_mass_tuner;
    std::unique_ptr<ApplyExternalForcesTuner> apply_external_forces_tuner;

    std::map<tuners, Tuner*> active_tuners;
    std::unique_ptr<TuningScheduler> scheduler;
    float tuning_budget;

    friend class GUI;
};
