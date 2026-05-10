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


class CUDASPHSimulator final : public FluidSimulator {
public:
    ///////////////////////////////////////////////////////////////////////////////
    ////                         SIMULATION PARAMETERS                         ////
    ///////////////////////////////////////////////////////////////////////////////
    static constexpr float REST_DENSITY = 1000.f;
    static constexpr float PARTICLE_VOLUME = PARTICLE_SPACING * PARTICLE_SPACING * PARTICLE_SPACING * 0.8;
    static constexpr float PARTICLE_MASS = REST_DENSITY * PARTICLE_VOLUME;

    static constexpr float SUPPORT_RADIUS = 2.f * PARTICLE_SPACING;
    static constexpr float CELL_SIZE = SUPPORT_RADIUS;

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

    void set_frozen_config(ktt::KernelConfiguration cfg);
    bool was_scheduled_step() const { return is_scheduled(STEP_TUNER); }
    std::pair<float, float> compute_state_metrics() const;

private:
    void init_boundary();
    void build_boundary_n_search(float4* positions_dev_ptr);
    bool has_boundary() const;
    void compute_boundary_mass(float4* positions_dev_ptr);

    // Adapt the time step size according to the Courant-Friedrich-Levy (CFL) condition
    float adapt_time_step(float delta, float min_step, float max_step) const;

    enum tuners {
        STEP_TUNER,
        UPDATE_POSITIONS_TUNER,
    };

    bool is_scheduled(tuners tuner_i) const;

    std::pair<int, int> tuning_stats() const;

    void set_tuning_budget(float tuning_budget);
    void reset_tuning();

    ParticleData particle_data;
    ParticleDataVisualizer particle_data_visualizer;

    std::unique_ptr<NSearchWrapperTuned> boundary_n_search;

    StepTuner step_tuner;

    std::unique_ptr<ComputeBoundaryMassTuner> compute_boundary_mass_tuner;

    std::map<tuners, Tuner*> active_tuners;
    std::unique_ptr<TuningScheduler> scheduler;
    float tuning_budget;

    friend class GUI;
    friend class Application;
};
