#include "sph.cuh"

#include "kernel.cuh"


namespace kernels {

__global__ void compute_pressure_k(const float* densities, float* pressures, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float d = fmaxf(densities[i], CUDASPHSimulator<>::REST_DENSITY);
    pressures[i] = CUDASPHSimulator<>::STIFFNESS *
        (powf(d / CUDASPHSimulator<>::REST_DENSITY, CUDASPHSimulator<>::EXPONENT) - 1.f);
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
//             float q = r_to_q(r, CUDASPHSimulator<>::SUPPORT_RADIUS);
//             float4 grad_W = spiky_grad(q, CUDASPHSimulator<>::SUPPORT_RADIUS) * r;
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
    float b_p_term = dpi + pressures[i] / (CUDASPHSimulator<>::REST_DENSITY * CUDASPHSimulator<>::REST_DENSITY);

    dev_n_search->for_neighbors(xi, [=, &p_accel, &boundary_p_accel] __device__ (unsigned j) {
        float4 xj = get_pos(positions, j);

        if (!is_neighbor(xi, xj, i, j))
            return;

        float4 r = xi - xj;
        float q = r_to_q(r, CUDASPHSimulator<>::SUPPORT_RADIUS);
        float4 grad_W = spiky_grad(q, CUDASPHSimulator<>::SUPPORT_RADIUS) * r;

        if (is_boundary(j, n)) {
            boundary_p_accel -= get_mass(boundary_mass, j, n) * b_p_term * grad_W;
        } else {
            float dj = densities[j];
            float dpj = pressures[j] / (dj * dj);
            p_accel -= (dpi + dpj) * grad_W;
        }
    });

    velocities[i] += delta * (CUDASPHSimulator<>::PARTICLE_MASS * p_accel + boundary_p_accel);
}

}
