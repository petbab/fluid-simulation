#pragma once
#include "sph_base.cuh"
#include "cuda/tuning/apply_pressure_force.cuh"
#include "cuda/tuning/compute_pressure.cuh"


template<ExternalForce ExtForce = no_force>
class CUDASPHSimulator final : public CUDASPHBase<ExtForce> {
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

    CUDASPHSimulator(const FluidSimulator::opts_t &opts)
    : CUDASPHBase<ExtForce>(opts), pressure(this->fluid_particles),
      compute_pressure_tuner(this->fluid_particles),
      apply_pressure_force_tuner(this->fluid_particles) {
    }

    void update(float delta) override {
        auto lock = this->cuda_gl_positions->lock();
        float* positions_ptr = static_cast<float*>(lock.get_ptr());

        this->n_search.rebuild(positions_ptr, this->total_particles);

        this->compute_boundary_mass(positions_ptr);

        this->compute_densities(positions_ptr);

        this->apply_non_pressure_forces(positions_ptr, delta);

        delta = this->adapt_time_step(delta, MIN_TIME_STEP, MAX_TIME_STEP);

        compute_pressure();
        apply_pressure_force(positions_ptr, delta);

        this->update_positions(positions_ptr, delta);
    }

    void visualize(Shader* shader) override {
        this->visualizer->visualize(shader, thrust::raw_pointer_cast(this->density.data()),
            CUDASPHBase<>::REST_DENSITY * 0.5f, CUDASPHBase<>::REST_DENSITY * 1.2f);
        // visualizer->visualize(shader, thrust::raw_pointer_cast(velocity.data()));
        // visualizer->visualize(shader, thrust::raw_pointer_cast(boundary_mass.data()),
        //     0.f, PARTICLE_MASS * 2.f, true);
    }

    void reset() override {
        CUDASPHBase<ExtForce>::reset();

        thrust::fill(pressure.begin(), pressure.end(), 0);
    }

private:
    void compute_pressure() {
        float* densities_ptr = thrust::raw_pointer_cast(this->density.data());
        float* pressures_ptr = thrust::raw_pointer_cast(pressure.data());
        compute_pressure_tuner.run(densities_ptr, pressures_ptr, this->total_particles);
    }

    void apply_pressure_force(float* positions_dev_ptr, float delta) {
        float* densities_ptr = thrust::raw_pointer_cast(this->density.data());
        float* pressures_ptr = thrust::raw_pointer_cast(pressure.data());
        float4* velocities_ptr = thrust::raw_pointer_cast(this->velocity.data());
        float* boundary_mass_ptr = thrust::raw_pointer_cast(this->boundary_mass.data());
        apply_pressure_force_tuner.run(
            positions_dev_ptr, densities_ptr, pressures_ptr, velocities_ptr,
            boundary_mass_ptr, this->fluid_particles, this->boundary_particles,
            delta, this->n_search.dev_ptr());
    }

    thrust::device_vector<float> pressure;
    ComputePressureTuner compute_pressure_tuner;
    ApplyPressureForceTuner apply_pressure_force_tuner;
};
