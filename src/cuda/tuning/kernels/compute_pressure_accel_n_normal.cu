#ifdef NOT_IN_KTT
#include "common.cuh"
#endif

#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


__global__ void compute_pressure_accel_n_normal(
    const float4* positions, const float* densities,
    const float* pressures, float4* accelerations, float4* normals,
    unsigned n, const NSearch *dev_n_search,
    const unsigned* nl_offsets,
    const unsigned* nl_counts,
    const unsigned* nl_neighbors
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float4 p_accel{0.f};
    float4 xi = positions[i];
    float di = densities[i];
    float dpi = pressures[i] / (di * di);

    float4 normal{0.f};

#if NEIGHBOR_LIST
    const unsigned start = nl_offsets[i];
    const unsigned end   = start + nl_counts[i];
    // #pragma unroll compute_non_pressure_accel_u_n
    for (unsigned k = start; k < end; ++k) {
        unsigned j = nl_neighbors[k];
#else
    dev_n_search->for_neighbors<compute_pressure_accel_n_normal_u_n>(xi, [=, &p_accel, &normal] (unsigned j) {
#endif
        float4 xj = positions[j];

#if !NEIGHBOR_LIST
        if (!is_neighbor(xi, xj, i, j))
            return;
#endif

        float4 r = xi - xj;
        float q = r_to_q(r, SUPPORT_RADIUS);
        float4 grad_W = spiky_grad(q, SUPPORT_RADIUS) * r;
        float dj = densities[j];
        float dpj = pressures[j] / (dj * dj);
        p_accel -= (dpi + dpj) * grad_W;

        normal += r * cubic_spline_grad(q, SUPPORT_RADIUS) / dj;
    }
#if !NEIGHBOR_LIST
    );
#endif

    accelerations[i] = PARTICLE_MASS * p_accel;

    normals[i] = SUPPORT_RADIUS * PARTICLE_MASS * normal;
}

__global__ void compute_pressure_accel_n_normal_with_boundary(
    const float4* positions, const float* densities,
    const float* pressures, float4* accelerations, float4* normals,
    unsigned fluid_n, const NSearch *fluid_n_search,
    const unsigned* nl_offsets,
    const unsigned* nl_counts,
    const unsigned* nl_neighbors,
    const float* boundary_mass, const NSearch *boundary_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fluid_n)
        return;

    float4 p_accel{0.f};
    float4 boundary_p_accel{0.f};
    float4 xi = positions[i];
    float di = densities[i];
    float dpi = pressures[i] / (di * di);
    float b_p_term = dpi + pressures[i] / (REST_DENSITY * REST_DENSITY);

    float4 normal{0.f};

#if NEIGHBOR_LIST
    const unsigned start = nl_offsets[i];
    const unsigned end   = start + nl_counts[i];
    // #pragma unroll compute_non_pressure_accel_u_n
    for (unsigned k = start; k < end; ++k) {
        unsigned j = nl_neighbors[k];
#else
    fluid_n_search->for_neighbors<compute_pressure_accel_n_normal_u_n>(xi, [=, &p_accel, &normal] (unsigned j) {
#endif
        float4 xj = positions[j];

#if !NEIGHBOR_LIST
        if (!is_neighbor(xi, xj, i, j))
            return;
#endif

        float4 r = xi - xj;
        float q = r_to_q(r, SUPPORT_RADIUS);
        float4 grad_W = spiky_grad(q, SUPPORT_RADIUS) * r;
        float dj = densities[j];
        float dpj = pressures[j] / (dj * dj);
        p_accel -= (dpi + dpj) * grad_W;

        normal += r * cubic_spline_grad(q, SUPPORT_RADIUS) / dj;
    }
#if !NEIGHBOR_LIST
);
#endif

    normals[i] = SUPPORT_RADIUS * PARTICLE_MASS * normal;

    boundary_n_search->for_boundary_neighbors(xi, [=, &boundary_p_accel] (unsigned j) {
        j += fluid_n;
        float4 xj = positions[j];
        if (!is_neighbor(xi, xj, i, j))
            return;

        float4 r = xi - xj;
        float q = r_to_q(r, SUPPORT_RADIUS);
        float4 grad_W = spiky_grad(q, SUPPORT_RADIUS) * r;
        boundary_p_accel -= get_mass(boundary_mass, j, fluid_n) * b_p_term * grad_W;
    });

    accelerations[i] = PARTICLE_MASS * p_accel + boundary_p_accel;
}
