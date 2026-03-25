#include "sph_base.cuh"
#include "kernel.cuh"


namespace kernels {

__global__ void update_velocities_k(float4* velocities, const float4* acceleration, unsigned n, float delta) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    velocities[i] += acceleration[i] * delta;
}

__global__ void compute_viscosity_k(
    const float* positions, const float4* velocities,
    const float* densities, float4* acceleration,
    unsigned n, const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float4 velocity_laplacian{0.f};
    float4 xi = get_pos(positions, i);
    float4 vi = velocities[i];

    dev_n_search->for_neighbors(xi, [=, &velocity_laplacian] __device__ (unsigned j) {
        if (is_boundary(j, n))
            return;

        float4 xj = get_pos(positions, j);

        if (is_neighbor(xi, xj, i, j)) {
            float4 x_ij = xi - xj;
            float4 v_ij = vi - velocities[j];
            float q = r_to_q(x_ij, CUDASPHBase<>::SUPPORT_RADIUS);

            velocity_laplacian += dot(v_ij, x_ij) * cubic_spline_grad(q, CUDASPHBase<>::SUPPORT_RADIUS)
                * x_ij / (densities[j] * (dot(x_ij, x_ij)
                    + 0.01f * CUDASPHBase<>::SUPPORT_RADIUS * CUDASPHBase<>::SUPPORT_RADIUS));
        }
    });

    velocity_laplacian *= 10 * CUDASPHBase<>::PARTICLE_MASS;
    acceleration[i] += CUDASPHBase<>::VISCOSITY * velocity_laplacian;
}

__global__ void compute_surface_tension_k(
    const float* positions, const float* densities,
    const float4* normals, float4* acceleration,
    unsigned n, const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float4 xi = get_pos(positions, i);
    float4 ni = normals[i];
    float di = densities[i];
    float4 f{0.f};

    dev_n_search->for_neighbors(xi, [=, &f] __device__ (unsigned j) {
        if (is_boundary(j, n))
            return;

        float4 xj = get_pos(positions, j);

        if (is_neighbor(xi, xj, i, j)) {
            float4 x_ij = xi - get_pos(positions, j);
            float q = r_to_q(x_ij, CUDASPHBase<>::SUPPORT_RADIUS);
            if (dot(x_ij, x_ij) > 1e-6)
                f += (CUDASPHBase<>::PARTICLE_MASS * normalize(x_ij)
                    * cohesion(q, CUDASPHBase<>::SUPPORT_RADIUS) + ni - normals[j]) / (di + densities[j]);
        }
    });

    acceleration[i] -= CUDASPHBase<>::SURFACE_TENSION_ALPHA * 2.f * CUDASPHBase<>::REST_DENSITY * f;
}

__global__ void compute_surface_normals_k(
    const float* positions, const float* densities,
    float4* normals, unsigned n,
    const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float4 xi = get_pos(positions, i);
    float4 normal{0.f};

    dev_n_search->for_neighbors(xi, [=, &normal] __device__ (unsigned j) {
        if (is_boundary(j, n))
            return;

        float4 xj = get_pos(positions, j);

        if (is_neighbor(xi, xj, i, j)) {
            float4 r = xi - xj;
            float q = r_to_q(r, CUDASPHBase<>::SUPPORT_RADIUS);
            normal += r * cubic_spline_grad(q, CUDASPHBase<>::SUPPORT_RADIUS) / densities[j];
        }
    });

    normals[i] = CUDASPHBase<>::SUPPORT_RADIUS * CUDASPHBase<>::PARTICLE_MASS * normal;
}

}
