#pragma once
#include "sph_base.cuh"


namespace kernels {

static constexpr dim3 COMPUTE_PRESSURE_BLOCK_SIZE = {128};
static constexpr dim3 APPLY_PRESSURE_FORCE_BLOCK_SIZE = {128};

__global__ void compute_pressure_k(const float* densities, float* pressures, unsigned n);
__global__ void apply_pressure_force_k(
    const float* positions, const float* densities,
    const float* pressures, float4* velocities,
    const float* boundary_mass,
    unsigned n, float delta, const NSearch *dev_n_search);

}

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
    : CUDASPHBase<ExtForce>(opts), pressure(this->fluid_particles) {
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
        const float* densities_ptr = thrust::raw_pointer_cast(this->density.data());
        float* pressures_ptr = thrust::raw_pointer_cast(pressure.data());

        const dim3 grid_size{this->fluid_particles / kernels::COMPUTE_PRESSURE_BLOCK_SIZE.x + 1};
        kernels::compute_pressure_k<<<kernels::COMPUTE_PRESSURE_BLOCK_SIZE, grid_size>>>(
            densities_ptr, pressures_ptr, this->fluid_particles);
        cudaCheckError();
    }

    void apply_pressure_force(const float* positions_dev_ptr, float delta) {
        const float* densities_ptr = thrust::raw_pointer_cast(this->density.data());
        const float* pressures_ptr = thrust::raw_pointer_cast(pressure.data());
        float4* velocities_ptr = thrust::raw_pointer_cast(this->velocity.data());
        const float* boundary_mass_ptr = thrust::raw_pointer_cast(this->boundary_mass.data());

        const dim3 grid_size{this->fluid_particles / kernels::APPLY_PRESSURE_FORCE_BLOCK_SIZE.x + 1};
        kernels::apply_pressure_force_k<<<kernels::APPLY_PRESSURE_FORCE_BLOCK_SIZE, grid_size>>>(
            positions_dev_ptr, densities_ptr, pressures_ptr, velocities_ptr, boundary_mass_ptr,
            this->fluid_particles, delta, this->n_search.dev_ptr());
        cudaCheckError();
    }

    thrust::device_vector<float> pressure;
};
