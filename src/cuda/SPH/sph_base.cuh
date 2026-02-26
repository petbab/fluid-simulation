#pragma once

#include <thrust/device_vector.h>
#include <cuda/simulator.h>
#include <cuda/nsearch/nsearch.h>
#include <cuda/math.cuh>
#include <thrust/transform_reduce.h>
#include <cuda/util.cuh>


template<class F>
concept ExternalForce = requires (F f, float4 pos, float4 acc) {
    acc = f(pos);
    acc += f(pos);
    acc -= f(pos);
} && DeviceCallable<F>;

struct no_force {
    __device__ float4 operator()(float4) const { return make_float4(0.f); }
};
static_assert(ExternalForce<no_force>);

static constexpr float4 GRAVITY{0, -9.81f, 0, 0};

namespace kernels {

static constexpr dim3 COMPUTE_DENSITY_BLOCK_SIZE = {128};
static constexpr dim3 UPDATE_POSITIONS_BLOCK_SIZE = {128};
static constexpr dim3 UPDATE_VELOCITIES_BLOCK_SIZE = {128};
static constexpr dim3 COMPUTE_XSPH_BLOCK_SIZE = {128};
static constexpr dim3 COMPUTE_VISCOSITY_BLOCK_SIZE = {128};
static constexpr dim3 COMPUTE_SURFACE_TENSION_BLOCK_SIZE = {128};
static constexpr dim3 COMPUTE_SURFACE_NORMALS_BLOCK_SIZE = {128};
static constexpr dim3 COMPUTE_BOUNDARY_MASS_BLOCK_SIZE = {128};
static constexpr dim3 APPLY_EXTERNAL_FORCES_BLOCK_SIZE = {128};

__global__ void compute_densities_k(
    const float* positions, float* densities,
    const float *boundary_mass,
    unsigned n, const NSearch *dev_n_search
);

__global__ void update_positions_k(float* positions, float4* velocities, unsigned n, float delta, BoundingBox bb);
__global__ void update_velocities_k(float4* velocities, const float4* acceleration, unsigned n, float delta);
__global__ void compute_viscosity_k(
    const float* positions, const float4* velocities,
    const float* densities, float4* acceleration,
    unsigned n, const NSearch *dev_n_search);
__global__ void compute_surface_tension_k(
    const float* positions, const float* densities,
    const float4* normals, float4* acceleration,
    unsigned n, const NSearch *dev_n_search);
__global__ void compute_surface_normals_k(
    const float* positions, const float* densities,
    float4* normals, unsigned n,
    const NSearch *dev_n_search);
__global__ void compute_boundary_mass_k(
    const float* positions, float* masses,
    unsigned fluid_n, unsigned boundary_n,
    const NSearch *dev_n_search);

template<ExternalForce EF>
__global__ void apply_external_forces_k(const float* positions, float4* acceleration, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        acceleration[i] = GRAVITY + EF{}(get_pos(positions, i));
}

}

template<ExternalForce ExtForce = no_force>
class CUDASPHBase : public CUDASimulator {
public:
    ///////////////////////////////////////////////////////////////////////////////
    ////                         SIMULATION PARAMETERS                         ////
    ///////////////////////////////////////////////////////////////////////////////
    static constexpr float REST_DENSITY = 1000.f;
    static constexpr float SUPPORT_RADIUS = 2.f * PARTICLE_SPACING;
    static constexpr float PARTICLE_VOLUME = PARTICLE_SPACING * PARTICLE_SPACING * PARTICLE_SPACING * 0.8;
    static constexpr float PARTICLE_MASS = REST_DENSITY * PARTICLE_VOLUME;

    static constexpr float ELASTICITY = 0.9f;


    static constexpr float XSPH_ALPHA = 0.f;
    static constexpr float VISCOSITY = 0.001f;
    static constexpr float SURFACE_TENSION_ALPHA = 0.15f;

    static constexpr float CFL_FACTOR = 0.4f;
    static constexpr float NON_PRESSURE_MAX_TIME_STEP = 0.015;
    ///////////////////////////////////////////////////////////////////////////////

    struct float4_length_sq {
        __host__ __device__ float operator()(const float4& v) const {
            return length(v);
        }
    };

    CUDASPHBase(const opts_t &opts)
    : CUDASimulator(opts),
      density(fluid_particles),
      boundary_mass(boundary_particles),
      velocity(fluid_particles),
      n_search{2.f * SUPPORT_RADIUS},
      non_pressure_accel(fluid_particles),
      normal(fluid_particles) {
    }

protected:
    void compute_densities(const float* positions_dev_ptr) {
        float* densities_ptr = thrust::raw_pointer_cast(density.data());
        float* boundary_mass_ptr = thrust::raw_pointer_cast(boundary_mass.data());

        const dim3 grid_size{fluid_particles / kernels::COMPUTE_DENSITY_BLOCK_SIZE.x + 1};
        kernels::compute_densities_k<<<kernels::COMPUTE_DENSITY_BLOCK_SIZE, grid_size>>>(
            positions_dev_ptr, densities_ptr, boundary_mass_ptr, fluid_particles, n_search.dev_ptr());
        cudaCheckError();
    }

    void update_positions(float* positions_dev_ptr, float delta) {
        float4* velocities_ptr = thrust::raw_pointer_cast(velocity.data());

        const dim3 grid_size{fluid_particles / kernels::UPDATE_POSITIONS_BLOCK_SIZE.x + 1};
        kernels::update_positions_k<<<kernels::UPDATE_POSITIONS_BLOCK_SIZE, grid_size>>>(
            positions_dev_ptr, velocities_ptr, fluid_particles, delta, bounding_box);
        cudaCheckError();
    }

    void apply_non_pressure_forces(const float* positions_dev_ptr, float delta) {
        apply_external_forces(positions_dev_ptr);

        compute_viscosity(positions_dev_ptr);
        compute_surface_tension(positions_dev_ptr);

        update_velocities(delta);

        compute_XSPH(positions_dev_ptr);
    }

    void compute_boundary_mass(const float* positions_dev_ptr) {
        float* masses_ptr = thrust::raw_pointer_cast(boundary_mass.data());

        const dim3 grid_size{boundary_particles / kernels::COMPUTE_BOUNDARY_MASS_BLOCK_SIZE.x + 1};
        kernels::compute_boundary_mass_k<<<kernels::COMPUTE_BOUNDARY_MASS_BLOCK_SIZE, grid_size>>>(
            positions_dev_ptr, masses_ptr, fluid_particles, boundary_particles, n_search.dev_ptr());
        cudaCheckError();
    }

    void reset() override {
        FluidSimulator::reset();

        thrust::fill(density.begin(), density.end(), 0);
        thrust::fill(boundary_mass.begin(), boundary_mass.end(), 0);
        thrust::fill(velocity.begin(), velocity.end(), make_float4(0));
        thrust::fill(non_pressure_accel.begin(), non_pressure_accel.end(), make_float4(0));
        thrust::fill(normal.begin(), normal.end(), make_float4(0));
    }

    // Adapt the time step size according to the Courant-Friedrich-Levy (CFL) condition
    float adapt_time_step(float delta, float min_step, float max_step) const {
        float max_velocity = sqrtf(thrust::transform_reduce(
            velocity.begin(),
            velocity.end(),
            float4_length_sq(),      // unary transform
            0.0f,                    // initial value
            thrust::maximum<float>() // reduction op
        ));

        if (max_velocity < 1.e-9)
            return max_step;

        float cfl_max_time_step = CFL_FACTOR * PARTICLE_SPACING / max_velocity;

        return std::min(std::clamp(delta, min_step, max_step), cfl_max_time_step);
    }

private:
    void update_velocities(float delta) {
        delta = std::min(delta, NON_PRESSURE_MAX_TIME_STEP);

        float4* velocities_ptr = thrust::raw_pointer_cast(velocity.data());
        const float4* acceleration_ptr = thrust::raw_pointer_cast(non_pressure_accel.data());

        const dim3 grid_size{fluid_particles / kernels::UPDATE_VELOCITIES_BLOCK_SIZE.x + 1};
        kernels::update_velocities_k<<<kernels::UPDATE_VELOCITIES_BLOCK_SIZE, grid_size>>>(
            velocities_ptr, acceleration_ptr, fluid_particles, delta);
        cudaCheckError();
    }

    /**
     * Simulates viscosity by smoothing the velocity field [SPH Tutorial, eq. 103].
     */
    void compute_XSPH(const float* positions_dev_ptr) {
        if constexpr (XSPH_ALPHA == 0.f)
            return;
        // TODO
    }

    /**
     * Computes and applies the viscous force using an explicit viscosity model.
     * Approximates the Laplacian of the velocity field via finite differences
     * [SPH Tutorial, eq. 102].
     */
    void compute_viscosity(const float* positions_dev_ptr) {
        if constexpr (VISCOSITY == 0.f)
            return;

        const float* densities_ptr = thrust::raw_pointer_cast(density.data());
        const float4* velocities_ptr = thrust::raw_pointer_cast(velocity.data());
        float4* acceleration_ptr = thrust::raw_pointer_cast(non_pressure_accel.data());

        const dim3 grid_size{fluid_particles / kernels::COMPUTE_VISCOSITY_BLOCK_SIZE.x + 1};
        kernels::compute_viscosity_k<<<kernels::COMPUTE_VISCOSITY_BLOCK_SIZE, grid_size>>>(
            positions_dev_ptr, velocities_ptr, densities_ptr,
            acceleration_ptr, fluid_particles, n_search.dev_ptr());
        cudaCheckError();
    }

    /**
     * Simulates surface tension using the macroscopic approach by Akinci et al. (2013).
     */
    void compute_surface_tension(const float* positions_dev_ptr) {
        compute_surface_normals(positions_dev_ptr);

        const float* densities_ptr = thrust::raw_pointer_cast(density.data());
        const float4* normals_ptr = thrust::raw_pointer_cast(normal.data());
        float4* acceleration_ptr = thrust::raw_pointer_cast(non_pressure_accel.data());

        const dim3 grid_size{fluid_particles / kernels::COMPUTE_SURFACE_TENSION_BLOCK_SIZE.x + 1};
        kernels::compute_surface_tension_k<<<kernels::COMPUTE_SURFACE_TENSION_BLOCK_SIZE, grid_size>>>(
            positions_dev_ptr, densities_ptr, normals_ptr,
            acceleration_ptr, fluid_particles, n_search.dev_ptr());
        cudaCheckError();
    }

    /**
     * Computes surface normals used to calculate the curvature force
     * in compute_surface_tension [SPH Tutorial, eq. 125].
     */
    void compute_surface_normals(const float* positions_dev_ptr) {
        const float* densities_ptr = thrust::raw_pointer_cast(density.data());
        float4* normals_ptr = thrust::raw_pointer_cast(normal.data());

        const dim3 grid_size{fluid_particles / kernels::COMPUTE_SURFACE_NORMALS_BLOCK_SIZE.x + 1};
        kernels::compute_surface_normals_k<<<kernels::COMPUTE_SURFACE_NORMALS_BLOCK_SIZE, grid_size>>>(
            positions_dev_ptr, densities_ptr, normals_ptr,
            fluid_particles, n_search.dev_ptr());
        cudaCheckError();
    }

    void apply_external_forces(const float* positions_dev_ptr) {
        if constexpr (!std::is_same_v<no_force, ExtForce>) {
            float4* acceleration_ptr = thrust::raw_pointer_cast(non_pressure_accel.data());

            const dim3 grid_size{fluid_particles / kernels::APPLY_EXTERNAL_FORCES_BLOCK_SIZE.x + 1};
            kernels::apply_external_forces_k<ExtForce><<<kernels::APPLY_EXTERNAL_FORCES_BLOCK_SIZE, grid_size>>>(
                positions_dev_ptr, acceleration_ptr, fluid_particles);
            cudaCheckError();
        } else {
            thrust::fill(non_pressure_accel.begin(), non_pressure_accel.end(), GRAVITY);
        }
    }

protected:
    thrust::device_vector<float> density, boundary_mass;
    thrust::device_vector<float4> velocity;
    NSearchWrapper n_search;

private:
    thrust::device_vector<float4> non_pressure_accel, normal;
};

__device__ __host__ inline bool is_neighbor(float4 xi, float4 xj, unsigned i, unsigned j) {
    float4 r = xi - xj;
    return i != j && dot(r, r) <= CUDASPHBase<>::SUPPORT_RADIUS * CUDASPHBase<>::SUPPORT_RADIUS;
}
