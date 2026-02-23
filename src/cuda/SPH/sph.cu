#include "sph.cuh"

#include "kernel.cuh"
#include <debug.h>

///////////////////////////////////////////////////////////////////////////////
////                        KERNEL HYPERPARAMETERS                         ////
///////////////////////////////////////////////////////////////////////////////

static constexpr dim3 COMPUTE_PRESSURE_BLOCK_SIZE = {128};
static constexpr dim3 APPLY_PRESSURE_FORCE_BLOCK_SIZE = {128};

///////////////////////////////////////////////////////////////////////////////
////                                KERNELS                                ////
///////////////////////////////////////////////////////////////////////////////

__global__ void compute_pressure_k(const float* densities, float* pressures, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float d = fmaxf(densities[i], CUDASPHSimulator::REST_DENSITY);
    pressures[i] = CUDASPHSimulator::STIFFNESS *
        (powf(d / CUDASPHSimulator::REST_DENSITY, CUDASPHSimulator::EXPONENT) - 1.f);
}

// https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf (Eq. 96)
// __device__ float compute_gamma_2(
//     unsigned i, float4 xi, const float* positions,
//     unsigned n, const NSearch *dev_n_search
// ) {
//     float4 fluid_grad_sum = make_float4(0.f);
//     float4 boundary_grad_sum = make_float4(0.f);
//
//     dev_n_search->for_neighbors(xi, [=, &fluid_grad_sum, &boundary_grad_sum] __device__ (unsigned j) {
//         float4 xj = get_pos(positions, j);
//
//         if (is_neighbor(xi, xj, i, j)) {
//             float4 r = xi - xj;
//             float q = r_to_q(r, CUDASPHSimulator::SUPPORT_RADIUS);
//             float4 grad_W = spiky_grad(q, CUDASPHSimulator::SUPPORT_RADIUS) * r;
//             if (is_boundary(j, n))
//                 boundary_grad_sum += grad_W;
//             else
//                 fluid_grad_sum += grad_W;
//         }
//     });
//
//     float denom = dot(boundary_grad_sum, boundary_grad_sum);
//     if (denom < 1.e-6f)
//         return 0.f;
//     return dot(fluid_grad_sum, boundary_grad_sum) / denom;
// }

__global__ void apply_pressure_force_k(
    const float* positions, const float* densities,
    const float* pressures, float4* velocities,
    const float* boundary_mass,
    unsigned n, float delta, const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float4 p_accel{0.f};
    float4 boundary_p_accel{0.f};
    float4 xi = get_pos(positions, i);
    float di = densities[i];
    float dpi = pressures[i] / (di * di);
    float b_p_term = dpi + pressures[i] / (CUDASPHSimulator::REST_DENSITY * CUDASPHSimulator::REST_DENSITY);

    dev_n_search->for_neighbors(xi, [=, &p_accel, &boundary_p_accel] __device__ (unsigned j) {
        float4 xj = get_pos(positions, j);

        if (!is_neighbor(xi, xj, i, j))
            return;

        float4 r = xi - xj;
        float q = r_to_q(r, CUDASPHSimulator::SUPPORT_RADIUS);
        float4 grad_W = spiky_grad(q, CUDASPHSimulator::SUPPORT_RADIUS) * r;

        if (is_boundary(j, n)) {
            boundary_p_accel -= get_mass(boundary_mass, j, n) * b_p_term * grad_W;
        } else {
            float dj = densities[j];
            float dpj = pressures[j] / (dj * dj);
            p_accel -= (dpi + dpj) * grad_W;
        }
    });

    velocities[i] += delta * (CUDASPHSimulator::PARTICLE_MASS * p_accel + boundary_p_accel);
}

///////////////////////////////////////////////////////////////////////////////
////                           CUDASPHSimulator                            ////
///////////////////////////////////////////////////////////////////////////////

CUDASPHSimulator::CUDASPHSimulator(const opts_t &opts)
    : CUDASPHBase(opts), pressure(fluid_particles) {
}

void CUDASPHSimulator::update(float delta) {
    auto lock = cuda_gl_positions->lock();
    float* positions_ptr = static_cast<float*>(lock.get_ptr());

    n_search.rebuild(positions_ptr, total_particles);

    compute_boundary_mass(positions_ptr);

    compute_densities(positions_ptr);

    apply_non_pressure_forces(positions_ptr, delta);

    delta = adapt_time_step(delta, MIN_TIME_STEP, MAX_TIME_STEP);

    compute_pressure();
    apply_pressure_force(positions_ptr, delta);

    update_positions(positions_ptr, delta);
}

void CUDASPHSimulator::visualize(Shader* shader) {
    visualizer->visualize(shader, thrust::raw_pointer_cast(density.data()),
        REST_DENSITY * 0.5f, REST_DENSITY * 1.2f);
    // visualizer->visualize(shader, thrust::raw_pointer_cast(velocity.data()));
    // visualizer->visualize(shader, thrust::raw_pointer_cast(boundary_mass.data()),
    //     0.f, PARTICLE_MASS * 2.f, true);
}

void CUDASPHSimulator::reset() {
    CUDASPHBase::reset();

    thrust::fill(pressure.begin(), pressure.end(), 0);
}

void CUDASPHSimulator::compute_pressure() {
    const float* densities_ptr = thrust::raw_pointer_cast(density.data());
    float* pressures_ptr = thrust::raw_pointer_cast(pressure.data());

    const dim3 grid_size{fluid_particles / COMPUTE_PRESSURE_BLOCK_SIZE.x + 1};
    compute_pressure_k<<<COMPUTE_PRESSURE_BLOCK_SIZE, grid_size>>>(
        densities_ptr, pressures_ptr, fluid_particles);
    cudaCheckError();
}

void CUDASPHSimulator::apply_pressure_force(const float* positions_dev_ptr, float delta) {
    const float* densities_ptr = thrust::raw_pointer_cast(density.data());
    const float* pressures_ptr = thrust::raw_pointer_cast(pressure.data());
    float4* velocities_ptr = thrust::raw_pointer_cast(velocity.data());
    const float* boundary_mass_ptr = thrust::raw_pointer_cast(boundary_mass.data());

    const dim3 grid_size{fluid_particles / APPLY_PRESSURE_FORCE_BLOCK_SIZE.x + 1};
    apply_pressure_force_k<<<APPLY_PRESSURE_FORCE_BLOCK_SIZE, grid_size>>>(
        positions_dev_ptr, densities_ptr, pressures_ptr, velocities_ptr, boundary_mass_ptr,
        fluid_particles, delta, n_search.dev_ptr());
    cudaCheckError();
}
