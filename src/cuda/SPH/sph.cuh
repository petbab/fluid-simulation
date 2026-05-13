#pragma once

#include <thrust/device_vector.h>
#include <cuda/nsearch/nsearch.h>
#include "cuda/tuning/compute_boundary_mass.cuh"
#include "cuda/tuning/tuning_scheduler.h"
#include <memory>
#include "particle_data.cuh"
#include "particle_data_visualizer.cuh"
#include "cuda/tuning/step_tuner.cuh"
#include "simulation/fluid_simulator.h"


/**
 * @brief CUDA-based SPH fluid simulator with auto-tuning support.
 *
 * Implements a predictive-corrective incompressible SPH solver using
 * KTT-tuned kernels for neighbor search, density/pressure computation,
 * force integration, and position updates.
 */
class CUDASPHSimulator final : public FluidSimulator {
public:
    ///////////////////////////////////////////////////////////////////////////////
    ////                         SIMULATION PARAMETERS                         //////
    ///////////////////////////////////////////////////////////////////////////////
    static constexpr float REST_DENSITY = 1000.f;  ///< Rest density (kg/m^3).
    static constexpr float PARTICLE_VOLUME = PARTICLE_SPACING * PARTICLE_SPACING * PARTICLE_SPACING * 0.8;  ///< Particle volume.
    static constexpr float PARTICLE_MASS = REST_DENSITY * PARTICLE_VOLUME;  ///< Particle mass.

    static constexpr float SUPPORT_RADIUS = 2.f * PARTICLE_SPACING;  ///< SPH support radius.
    static constexpr float CELL_SIZE = SUPPORT_RADIUS;               ///< Spatial hash cell size.

    static constexpr float4 GRAVITY{0, -9.81f, 0, 0};  ///< Gravitational acceleration.

    static constexpr float CFL_FACTOR = 0.4f;                    ///< CFL stability factor.
    static constexpr float NON_PRESSURE_MAX_TIME_STEP = 0.015;   ///< Max non-pressure time step.

    static constexpr float MAX_TIME_STEP = 0.0005f;  ///< Maximum simulation time step.
    static constexpr float MIN_TIME_STEP = 0.00001f; ///< Minimum simulation time step.
    ///////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Constructs the simulator.
     * @param opts Simulation options (origin, grid, bounding box, etc.).
     */
    CUDASPHSimulator(const opts_t& opts);

    /**
     * @brief Initializes double-buffered position VBOs for CUDA interop.
     * @param pos_vao_a OpenGL buffer A.
     * @param pos_vao_b OpenGL buffer B.
     */
    void init_positions(GLuint pos_vao_a, GLuint pos_vao_b);

    /**
     * @brief Advances the simulation by one step.
     * @param delta Time step in seconds.
     */
    void update(float delta) override;

    /**
     * @brief Configures shader uniforms for particle visualization.
     * @param shader Shader to configure.
     */
    void visualize(Shader* shader) override;

    /** @brief Resets the simulation to initial state. */
    void reset() override;

private:
    void init_boundary();
    void build_boundary_n_search(float4* positions_dev_ptr);
    bool has_boundary() const;
    void compute_boundary_mass(float4* positions_dev_ptr);

    /**
     * @brief Adapts the time step size according to the CFL condition.
     * @param delta Requested time step.
     * @param min_step Minimum allowed time step.
     * @param max_step Maximum allowed time step.
     * @return Adapted time step.
     */
    float adapt_time_step(float delta, float min_step, float max_step) const;

    /** @brief Indices of tunable kernels. */
    enum tuners {
        STEP_TUNER,
        UPDATE_POSITIONS_TUNER,
    };

    /** @brief Checks if a tuner is scheduled for tuning this frame. */
    bool is_scheduled(tuners tuner_i) const;

    /** @brief Returns tuning statistics (searched, total). */
    std::pair<int, int> tuning_stats() const;

    void set_tuning_budget(float tuning_budget);
    void reset_tuning();

    ParticleData particle_data;                          ///< GPU particle data.
    ParticleDataVisualizer particle_data_visualizer;     ///< Visualization helper.

    std::unique_ptr<NSearchWrapperTuned> boundary_n_search;  ///< Boundary neighbor search.

    StepTuner step_tuner;                                ///< Main simulation step tuner.

    std::unique_ptr<ComputeBoundaryMassTuner> compute_boundary_mass_tuner;  ///< Boundary mass tuner.

    std::map<tuners, Tuner*> active_tuners;              ///< Map of active tuners.
    std::unique_ptr<TuningScheduler> scheduler;          ///< Tuning schedule manager.
    float tuning_budget;                                 ///< Fraction of frames spent tuning.

    friend class GUI;
};
