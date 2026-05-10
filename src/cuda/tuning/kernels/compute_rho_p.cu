#ifdef NOT_IN_KTT
#include "common.cuh"
#endif

#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


static constexpr float STIFFNESS = 0.1f;
static constexpr float EXPONENT = 7.f;
static constexpr float MAX_DENSITY_RATIO = 1.5f;


__forceinline__ __device__
float compute_density(
    const float4* positions,
    unsigned n, const NSearch *dev_n_search,
    const unsigned* nl_offsets,
    const unsigned* nl_counts,
    const unsigned* nl_neighbors,
    unsigned tile = 0
) {
#if compute_rho_p_nlist
#if compute_rho_p_shared
    constexpr unsigned BS    = compute_rho_p_block;
    constexpr unsigned CACHE = BS << (2 + compute_rho_p_cache_log2);
    constexpr unsigned MASK  = CACHE - 1;
    constexpr unsigned EMPTY = ~0u;

    __shared__ unsigned s_keys[CACHE];
    __shared__ float4   s_xj  [CACHE];

    const unsigned tid    = threadIdx.x;
    const unsigned i      = blockIdx.x * BS + tid;
    const bool     active = i < n;
    const unsigned i_safe = active ? i : 0u;

    const float4 xi = positions [i_safe];

    const unsigned start = active ? nl_offsets[i] : 0u;
    const unsigned count = active ? nl_counts [i] : 0u;

    for (unsigned s = tid; s < CACHE; s += BS) s_keys[s] = EMPTY;
    __syncthreads();

    const auto hash = [] (unsigned j) -> unsigned { return (j * 2654435761u) & MASK; };

    // Dedup-insert, first inserter performs the gather.
    for (unsigned k = 0; k < count; ++k) {
        const unsigned j = nl_neighbors[start + k];
        unsigned slot = hash(j);

        for (unsigned c = 0; c < CACHE; ++c) {
            unsigned cur = s_keys[slot];
            if (cur == j) break;
            if (cur == EMPTY) {
                unsigned old = atomicCAS(&s_keys[slot], EMPTY, j);
                if (old == EMPTY) {
                    s_xj[slot] = positions [j];
                    break;
                }
                if (old == j) break;
            }
            slot = (slot + 1) & MASK;
        }
    }
    __syncthreads();

    // Shared memory only consumption.
    float density = cubic_spline(0.f, SUPPORT_RADIUS);
    for (unsigned k = 0; k < count; ++k) {
        const unsigned j = nl_neighbors[start + k];
        unsigned slot = hash(j);
        for (int c = 0; s_keys[slot] != j && c < CACHE; ++c) slot = (slot + 1) & MASK;

        const float4 xj = s_xj[slot];
        float q = r_to_q(xi - xj, SUPPORT_RADIUS);
        density += cubic_spline(q, SUPPORT_RADIUS);
    }

    return density * PARTICLE_MASS;
#else
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return 0.f;

    float4 xi = positions[i];
    float density = cubic_spline(0.f, SUPPORT_RADIUS);

    const unsigned start = nl_offsets[i];
    const unsigned end   = start + nl_counts[i];
    #pragma unroll compute_rho_p_u_n
    for (unsigned k = start; k < end; ++k) {
        unsigned j = nl_neighbors[k];
        const float4 xj = positions[j];
        float q = r_to_q(xi - xj, SUPPORT_RADIUS);
        density += cubic_spline(q, SUPPORT_RADIUS);
    }

    return density * PARTICLE_MASS;
#endif
#else
#if compute_rho_p_shared
    constexpr unsigned BS = compute_rho_p_block;

    const unsigned t_i = blockIdx.x;
    const unsigned tid = threadIdx.x;

    if (dev_n_search->table[t_i] == NSearch::EMPTY_HASH) return 0.f;
    const unsigned home_start = dev_n_search->cell_start[t_i];
    const unsigned home_end   = dev_n_search->cell_end  [t_i];
    const unsigned home_count = home_end - home_start;
    if (home_start >= home_end) return 0.f;

    const unsigned in_cell_offset = BS * tile + threadIdx.x;
    const bool active = in_cell_offset < home_count;
    const unsigned i = active ? home_start + in_cell_offset : home_start;

    const float4 xi = positions[i];

    const NSearch::cell_t anchor = NSearch::cell_coord(xi);

    __shared__ float4   s_pos[BS];
    __shared__ unsigned s_idx[BS];

    float density = cubic_spline(0.f, SUPPORT_RADIUS);

    constexpr int R = static_cast<int>(SUPPORT_RADIUS / CELL_SIZE + .5f);

    for (int x = -R; x <= R; ++x)
    for (int y = -R; y <= R; ++y)
    for (int z = -R; z <= R; ++z) {
        NSearch::cell_t nc{anchor.x + x, anchor.y + y, anchor.z + z};
        unsigned n_t_i = dev_n_search->find_cell_in_table<TABLE_SIZE>(nc);
        if (n_t_i == NSearch::EMPTY_CELL) continue;

        const unsigned start = dev_n_search->cell_start[n_t_i];
        const unsigned end   = dev_n_search->cell_end  [n_t_i];
        const unsigned cnt   = end - start;
        if (start >= end) continue;

        // Cooperative load
        for (unsigned base = 0; base < cnt; base += BS) {
            const unsigned tile_n = min(BS, cnt - base);

            if (tid < tile_n) {
                unsigned j = start + base + tid;
                s_pos[tid] = positions[j];
                s_idx[tid] = j;
            }
            __syncthreads();

            // Each home thread consumes the cell.
            if (active) {
                for (unsigned k = 0; k < tile_n; ++k) {
                    if (i == s_idx[k]) continue;
                    float4 x_ij = xi - s_pos[k];
                    float r2 = dot(x_ij, x_ij);
                    if (r2 > SUPPORT_RADIUS * SUPPORT_RADIUS) continue;

                    float q = r_to_q(x_ij, SUPPORT_RADIUS);
                    density += cubic_spline(q, SUPPORT_RADIUS);
                }
            }
            __syncthreads();
        }
    }

    return density * PARTICLE_MASS;
#else
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return 0.f;

    float4 xi = positions[i];
    float density = cubic_spline(0.f, SUPPORT_RADIUS);

    dev_n_search->for_neighbors<compute_rho_p_u_n>(xi, [=, &density] (unsigned j) {
        float4 xj = positions[j];
        if (!is_neighbor(xi, xj, i, j))
            return;

        float4 x_ij = xi - xj;
        float q  = r_to_q(x_ij, SUPPORT_RADIUS);
        density += cubic_spline(q, SUPPORT_RADIUS);
    });

    return density * PARTICLE_MASS;
#endif
#endif
}

__global__ __launch_bounds__(compute_rho_p_block)
void compute_rho_p(
    const float4* positions, float* densities, float* pressures,
    unsigned n, const NSearch *dev_n_search,
    const unsigned* nl_offsets,
    const unsigned* nl_counts,
    const unsigned* nl_neighbors
) {
#if compute_rho_p_shared && !compute_rho_p_nlist
    const unsigned t_i = blockIdx.x;

    if (dev_n_search->table[t_i] == NSearch::EMPTY_HASH) return;
    const unsigned home_start = dev_n_search->cell_start[t_i];
    const unsigned home_end   = dev_n_search->cell_end  [t_i];
    const unsigned home_count = home_end - home_start;
    if (home_start >= home_end) return;

    constexpr unsigned BS = compute_rho_p_block;
    const unsigned tiles = (home_count + BS - 1) / BS;
    for (unsigned t = 0; t < tiles; ++t) {
        float density = compute_density(
            positions, n, dev_n_search, nl_offsets, nl_counts, nl_neighbors, t);

        const unsigned tid = BS * t + threadIdx.x;
        const bool active = tid < home_count;
        if (!active)
            return;

        const unsigned i = home_start + tid;
        densities[i] = density;

        float ratio = fmaxf(density / REST_DENSITY, 1.0f);
        ratio = fminf(ratio, MAX_DENSITY_RATIO);
        pressures[i] = STIFFNESS * (powf(ratio, EXPONENT) - 1.f);
    }
#else
    float density = compute_density(
        positions, n, dev_n_search, nl_offsets, nl_counts, nl_neighbors);

    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    densities[i] = density;

    float ratio = fmaxf(density / REST_DENSITY, 1.0f);
    ratio = fminf(ratio, MAX_DENSITY_RATIO);
    pressures[i] = STIFFNESS * (powf(ratio, EXPONENT) - 1.f);
#endif
}

__global__ __launch_bounds__(compute_rho_p_block)
void compute_rho_p_with_boundary(
    const float4* positions, float* densities, float* pressures,
    unsigned fluid_n, const NSearch *fluid_n_search,
    const unsigned* nl_offsets,
    const unsigned* nl_counts,
    const unsigned* nl_neighbors,
    const float *boundary_mass, const NSearch *boundary_n_search
) {
#if compute_rho_p_shared && !compute_rho_p_nlist
    const unsigned t_i = blockIdx.x;

    if (fluid_n_search->table[t_i] == NSearch::EMPTY_HASH) return;
    const unsigned home_start = fluid_n_search->cell_start[t_i];
    const unsigned home_end   = fluid_n_search->cell_end  [t_i];
    const unsigned home_count = home_end - home_start;
    if (home_start >= home_end) return;

    constexpr unsigned BS = compute_rho_p_block;
    const unsigned tiles = (home_count + BS - 1) / BS;
    for (unsigned t = 0; t < tiles; ++t) {
        float density = compute_density(
        positions, fluid_n, fluid_n_search, nl_offsets, nl_counts, nl_neighbors);

        const unsigned tid = BS * t + threadIdx.x;
        const bool active = tid < home_count;
        if (!active)
            return;

        const unsigned i = home_start + tid;
        float4 xi = positions[i];

        boundary_n_search->for_boundary_neighbors(xi, [=, &density] (unsigned j) {
            j += fluid_n;
            float4 xj = positions[j];

            if (is_neighbor(xi, xj, i, j)) {
                float q = r_to_q(xi - xj, SUPPORT_RADIUS);
                float W = cubic_spline(q, SUPPORT_RADIUS);
                density += W * get_mass(boundary_mass, j, fluid_n);
            }
        });

        densities[i] = density;

        float ratio = fmaxf(density / REST_DENSITY, 1.0f);
        ratio = fminf(ratio, MAX_DENSITY_RATIO);
        pressures[i] = STIFFNESS * (powf(ratio, EXPONENT) - 1.f);
    }
#else
    float density = compute_density(
        positions, fluid_n, fluid_n_search, nl_offsets, nl_counts, nl_neighbors);

    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fluid_n)
        return;

    float4 xi = positions[i];

    boundary_n_search->for_boundary_neighbors(xi, [=, &density] (unsigned j) {
        j += fluid_n;
        float4 xj = positions[j];

        if (is_neighbor(xi, xj, i, j)) {
            float q = r_to_q(xi - xj, SUPPORT_RADIUS);
            float W = cubic_spline(q, SUPPORT_RADIUS);
            density += W * get_mass(boundary_mass, j, fluid_n);
        }
    });

    densities[i] = density;

    float ratio = fmaxf(density / REST_DENSITY, 1.0f);
    ratio = fminf(ratio, MAX_DENSITY_RATIO);
    pressures[i] = STIFFNESS * (powf(ratio, EXPONENT) - 1.f);
#endif
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
