#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


static constexpr float STIFFNESS = 0.1f;
static constexpr float EXPONENT = 7.f;
static constexpr float MAX_DENSITY_RATIO = 1.5f;

__global__ void compute_rho_p(const float4* positions, float* densities, float* pressures,
    unsigned n, const NSearch *dev_n_search) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float4 xi = positions[i];
    float density = cubic_spline(0.f, SUPPORT_RADIUS);
    dev_n_search->for_neighbors<
        compute_rho_p_u_x,
        compute_rho_p_u_y,
        compute_rho_p_u_z
    >(xi, [=, &density] (unsigned j) {
        float4 xj = positions[j];

        if (is_neighbor(xi, xj, i, j)) {
            float q = r_to_q(xi - xj, SUPPORT_RADIUS);
            density += cubic_spline(q, SUPPORT_RADIUS);
        }
    });
    density *= PARTICLE_MASS;
    densities[i] = density;

    float ratio = fmaxf(density / REST_DENSITY, 1.0f);
    ratio = fminf(ratio, MAX_DENSITY_RATIO);
    pressures[i] = STIFFNESS * (powf(ratio, EXPONENT) - 1.f);
}

__global__ void compute_rho_p_with_boundary(
    const float4* positions, float* densities, float* pressures,
    unsigned fluid_n, const NSearch *fluid_n_search,
    const float *boundary_mass, const NSearch *boundary_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fluid_n)
        return;

    float4 xi = positions[i];
    float density = cubic_spline(0.f, SUPPORT_RADIUS);
    float boundary_density = 0.f;

    fluid_n_search->for_neighbors<
        compute_rho_p_u_x,
        compute_rho_p_u_y,
        compute_rho_p_u_z
    >(xi, [=, &density] (unsigned j) {
        float4 xj = positions[j];

        if (is_neighbor(xi, xj, i, j)) {
            float q = r_to_q(xi - xj, SUPPORT_RADIUS);
            float W = cubic_spline(q, SUPPORT_RADIUS);
            density += W;
        }
    });

    boundary_n_search->for_boundary_neighbors(xi, [=, &boundary_density] (unsigned j) {
        j += fluid_n;
        float4 xj = positions[j];

        if (is_neighbor(xi, xj, i, j)) {
            float q = r_to_q(xi - xj, SUPPORT_RADIUS);
            float W = cubic_spline(q, SUPPORT_RADIUS);
            boundary_density += W * get_mass(boundary_mass, j, fluid_n);
        }
    });

    density = density * PARTICLE_MASS + boundary_density;
    densities[i] = density;

    float ratio = fmaxf(density / REST_DENSITY, 1.0f);
    ratio = fminf(ratio, MAX_DENSITY_RATIO);
    pressures[i] = STIFFNESS * (powf(ratio, EXPONENT) - 1.f);
}

// __global__ void compute_densities_k(const float* positions, float* densities,
//                                      unsigned n, float support_radius, float particle_mass) {
//     unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
//
// #if GRID_STRIDE
//     for (; i < n; i += blockDim.x * gridDim.x) {
// #else
//     if (i < n) {
// #endif
//         float4 xi = positions[i];
//         float density = 0.f;
//
// #if USE_SHARED_MEMORY
//         __shared__ float s_positions[BLOCK_SIZE * 3];
// #endif
//
// #if UNROLL_FACTOR > 1
//         unsigned j = 0;
//         for (; j + UNROLL_FACTOR <= n; j += UNROLL_FACTOR) {
// #pragma unroll
//             for (int k = 0; k < UNROLL_FACTOR; ++k) {
//                 float4 xj = get_pos(positions, j + k);
//                 float q = r_to_q(xi - xj, support_radius);
//                 density += cubic_spline(q, support_radius);
//             }
//         }
//         // Handle remainder
//         for (; j < n; ++j) {
//             float4 xj = positions[j];
//             float q = r_to_q(xi - xj, support_radius);
//             density += cubic_spline(q, support_radius);
//         }
// #else
//         for (unsigned j = 0; j < n; ++j) {
//             float4 xj = positions[j];
//             float q = r_to_q(xi - xj, support_radius);
//             density += cubic_spline(q, support_radius);
//         }
// #endif
//
//         densities[i] = density * particle_mass;
// #if GRID_STRIDE
//     }
// #else
//     }
// #endif
// }
