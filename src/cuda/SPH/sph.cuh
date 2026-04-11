#pragma once
#include "sph_base.cuh"
#include "cuda/tuning/apply_pressure_force.cuh"
#include "cuda/tuning/compute_pressure.cuh"


class CUDASPHSimulator final : public CUDASPHBase {
public:
    ///////////////////////////////////////////////////////////////////////////////
    ////                         SIMULATION PARAMETERS                         ////
    ///////////////////////////////////////////////////////////////////////////////
    // static constexpr float STIFFNESS = 1.f;
    // static constexpr float EXPONENT = 3.f;
    static constexpr float STIFFNESS = 0.1f;
    static constexpr float EXPONENT = 7.f;

    static constexpr float MAX_TIME_STEP = 0.0005f;
    static constexpr float MIN_TIME_STEP = 0.00001f;
    ///////////////////////////////////////////////////////////////////////////////

    CUDASPHSimulator(const opts_t &opts)
    : CUDASPHBase(opts), pressure(fluid_particles),
      compute_pressure_tuner(fluid_particles, thrust::raw_pointer_cast(density.data()),
          thrust::raw_pointer_cast(pressure.data())),
      apply_pressure_force_tuner(fluid_particles, boundary_particles,
          thrust::raw_pointer_cast(density.data()), thrust::raw_pointer_cast(pressure.data()),
          thrust::raw_pointer_cast(velocity.data()), thrust::raw_pointer_cast(boundary_mass.data()),
          n_search.dev_ptr()) {
    }

    void update(float delta) override {
        auto lock = cuda_gl_positions->lock();
        float* positions_ptr = static_cast<float*>(lock.get_ptr());

        n_search.rebuild(positions_ptr);

        compute_boundary_mass(positions_ptr);

        compute_densities(positions_ptr);

        apply_non_pressure_forces(positions_ptr, delta);

        delta = adapt_time_step(delta, MIN_TIME_STEP, MAX_TIME_STEP);

        compute_pressure();
        apply_pressure_force(positions_ptr, delta);

        update_positions(positions_ptr, delta);
    }

    void visualize(Shader* shader) override {
        visualizer->visualize(shader, thrust::raw_pointer_cast(density.data()),
            REST_DENSITY * 0.5f, REST_DENSITY * 1.2f);
        // visualizer->visualize(shader, thrust::raw_pointer_cast(velocity.data()));
        // visualizer->visualize(shader, thrust::raw_pointer_cast(boundary_mass.data()),
        //     0.f, PARTICLE_MASS * 2.f, true);
    }

    void reset() override {
        CUDASPHBase::reset();

        thrust::fill(pressure.begin(), pressure.end(), 0);
    }

private:
    void compute_pressure() {
        compute_pressure_tuner.run();
    }

    void apply_pressure_force(float* positions_dev_ptr, float delta) {
        apply_pressure_force_tuner.run(positions_dev_ptr, delta);
    }

    thrust::device_vector<float> pressure;
    ComputePressureTuner compute_pressure_tuner;
    ApplyPressureForceTuner apply_pressure_force_tuner;
};
