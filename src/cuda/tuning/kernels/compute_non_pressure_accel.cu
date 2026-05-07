#ifdef NOT_IN_KTT
#include "common.cuh"
#endif

#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)

#ifndef EXTERNAL_FORCE
#define EXTERNAL_FORCE ([](float4 pos) { return make_float4(0., 0., 0., 0.); })
#endif


static constexpr float4 GRAVITY{0, -9.81f, 0, 0};
static constexpr float VISCOSITY = 0.001f;

static constexpr float SURFACE_TENSION_ALPHA = 0.13f;

__global__ void compute_non_pressure_accel(
    const float4* positions, const float* densities,
    const float4* velocities, const float4* normals, float4* acceleration,
    unsigned n, const NSearch *dev_n_search,
    const unsigned* nl_offsets,
    const unsigned* nl_counts,
    const unsigned* nl_neighbors
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 xi = positions[i];
    float4 vi = velocities[i];
    float4 ni = normals[i];
    float  di = densities[i];
    float4 f{0.f}, vlap{0.f};

#if NEIGHBOR_LIST
    const unsigned start = nl_offsets[i];
    const unsigned end   = start + nl_counts[i];
    // #pragma unroll compute_non_pressure_accel_u_n
    for (unsigned k = start; k < end; ++k) {
        unsigned j = nl_neighbors[k];

#else
    dev_n_search->for_neighbors<compute_non_pressure_accel_u_n>(xi, [=, &f, &vlap] (unsigned j) {
#endif
        float4 xj = positions[j];

#if !NEIGHBOR_LIST
        if (!is_neighbor(xi, xj, i, j))
            return;
#endif

        float4 x_ij = xi - xj;
        float r2 = dot(x_ij, x_ij);
        float dj = densities[j];
        float q  = r_to_q(x_ij, SUPPORT_RADIUS);

        if (r2 > 1e-6f)
            f += (PARTICLE_MASS * normalize(x_ij) * cohesion(q, SUPPORT_RADIUS)
                  + ni - normals[j]) / (di + dj);

        float4 v_ij = vi - velocities[j];
        vlap += dot(v_ij, x_ij) * cubic_spline_grad(q, SUPPORT_RADIUS) * x_ij
              / (dj * (r2 + 0.01f * SUPPORT_RADIUS * SUPPORT_RADIUS));
    }
#if !NEIGHBOR_LIST
    );
#endif

    acceleration[i] = GRAVITY + EXTERNAL_FORCE(xi)
        + VISCOSITY * 10.f * PARTICLE_MASS * vlap
        - SURFACE_TENSION_ALPHA * 2.f * REST_DENSITY * f;
}
