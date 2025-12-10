#define KERNEL_DIR /home/pbabic/Repositories/fluid-simulation/src/cuda/tuning/kernels
#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cu)


__global__ void compute_densities(const float* positions, float* densities, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    vec3 xi = get_pos(positions, i);
    float density = 0.f;

#if UNROLL_FACTOR > 1
    unsigned j = 0;
    for (; j + UNROLL_FACTOR <= n; j += UNROLL_FACTOR) {
        #pragma unroll
        for (int k = 0; k < UNROLL_FACTOR; ++k) {
            vec3 xj = get_pos(positions, j + k);
            float q = r_to_q(xi - xj, SUPPORT_RADIUS);
            density += cubic_spline(q, SUPPORT_RADIUS);
        }
    }
    // Handle remainder
    for (; j < n; ++j) {
        vec3 xj = get_pos(positions, j);
        float q = r_to_q(xi - xj, SUPPORT_RADIUS);
        density += cubic_spline(q, SUPPORT_RADIUS);
    }
#else
    for (unsigned j = 0; j < n; ++j) {
        vec3 xj = get_pos(positions, j);
        float q = r_to_q(xi - xj, SUPPORT_RADIUS);
        density += cubic_spline(q, SUPPORT_RADIUS);
    }
#endif
    densities[i] = density * PARTICLE_MASS;
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
//         vec3 xi = get_pos(positions, i);
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
//                 vec3 xj = get_pos(positions, j + k);
//                 float q = r_to_q(xi - xj, support_radius);
//                 density += cubic_spline(q, support_radius);
//             }
//         }
//         // Handle remainder
//         for (; j < n; ++j) {
//             vec3 xj = get_pos(positions, j);
//             float q = r_to_q(xi - xj, support_radius);
//             density += cubic_spline(q, support_radius);
//         }
// #else
//         for (unsigned j = 0; j < n; ++j) {
//             vec3 xj = get_pos(positions, j);
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
