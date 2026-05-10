#ifdef NOT_IN_KTT
#include "common.cuh"
#endif

#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


// Returns pressure acceleration
__forceinline__ __device__
float4 compute_pressure_accel_n_normal_impl(
    const float4* positions, const float* densities,
    const float* pressures, float4* normals,
    unsigned n, const NSearch *dev_n_search,
    const unsigned* nl_offsets,
    const unsigned* nl_counts,
    const unsigned* nl_neighbors,
    unsigned tile = 0
) {
#if compute_pressure_accel_n_normal_nlist
#if compute_pressure_accel_n_normal_shared
    constexpr unsigned BS    = compute_pressure_accel_n_normal_block;
    constexpr unsigned CACHE = BS << (2 + compute_pressure_accel_n_normal_cache_log2);
    constexpr unsigned MASK  = CACHE - 1;
    constexpr unsigned EMPTY = ~0u;

    __shared__ unsigned s_keys[CACHE];
    __shared__ float4   s_xj  [CACHE];
    __shared__ float    s_pj  [CACHE];
    __shared__ float    s_dj  [CACHE];

    const unsigned tid    = threadIdx.x;
    const unsigned i      = blockIdx.x * BS + tid;
    const bool     active = i < n;
    const unsigned i_safe = active ? i : 0u;

    const float4 xi = positions[i_safe];
    const float  di = densities[i_safe];
    const float dpi = pressures[i_safe] / (di * di);

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
                    s_xj[slot] = positions[j];
                    s_pj[slot] = pressures[j];
                    s_dj[slot] = densities[j];
                    break;
                }
                if (old == j) break;
            }
            slot = (slot + 1) & MASK;
        }
    }
    __syncthreads();

    // Shared memory only consumption.
    float4 p_accel{0.f}, normal{0.f};
    for (unsigned k = 0; k < count; ++k) {
        const unsigned j = nl_neighbors[start + k];
        unsigned slot = hash(j);
        for (int c = 0; s_keys[slot] != j && c < CACHE; ++c) slot = (slot + 1) & MASK;

        const float4 xj = s_xj[slot];
        const float  pj = s_pj[slot];
        const float  dj = s_dj[slot];

        const float4 x_ij = xi - xj;
        const float  q    = r_to_q(x_ij, SUPPORT_RADIUS);

        float4 grad_W = spiky_grad(q, SUPPORT_RADIUS) * x_ij;
        float dpj = pj / (dj * dj);
        p_accel -= (dpi + dpj) * grad_W;

        normal += x_ij * cubic_spline_grad(q, SUPPORT_RADIUS) / dj;
    }

    if (active)
        normals[i] = SUPPORT_RADIUS * PARTICLE_MASS * normal;
    return PARTICLE_MASS * p_accel;

#else
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return {0.f};

    float4 xi = positions[i];
    float  di = densities[i];
    float dpi = pressures[i] / (di * di);
    float4 p_accel{0.f}, normal{0.f};

    const unsigned start = nl_offsets[i];
    const unsigned end   = start + nl_counts[i];
    #pragma unroll compute_pressure_accel_n_normal_u_n
    for (unsigned k = start; k < end; ++k) {
        unsigned j = nl_neighbors[k];
        float4 r = xi - positions[j];
        float q = r_to_q(r, SUPPORT_RADIUS);
        float4 grad_W = spiky_grad(q, SUPPORT_RADIUS) * r;
        float dj = densities[j];
        float dpj = pressures[j] / (dj * dj);
        p_accel -= (dpi + dpj) * grad_W;

        normal += r * cubic_spline_grad(q, SUPPORT_RADIUS) / dj;
    }

    normals[i] = SUPPORT_RADIUS * PARTICLE_MASS * normal;
    return PARTICLE_MASS * p_accel;
#endif
#else
#if compute_pressure_accel_n_normal_shared
    constexpr unsigned BS = compute_pressure_accel_n_normal_block;

    const unsigned t_i = blockIdx.x;
    const unsigned tid = threadIdx.x;

    if (dev_n_search->table[t_i] == NSearch::EMPTY_HASH) return {0.f};
    const unsigned home_start = dev_n_search->cell_start[t_i];
    const unsigned home_end   = dev_n_search->cell_end  [t_i];
    const unsigned home_count = home_end - home_start;
    if (home_start >= home_end) return {0.f};

    const unsigned in_cell_offset = BS * tile + threadIdx.x;
    const bool active = in_cell_offset < home_count;
    const unsigned i = active ? home_start + in_cell_offset : home_start;

    const float4 xi = positions [i];
    const float  di = densities [i];
    const float dpi = pressures[i] / (di * di);

    const NSearch::cell_t anchor = NSearch::cell_coord(xi);

    __shared__ float4   s_pos [BS];
    __shared__ float    s_pres[BS];
    __shared__ float    s_den [BS];
    __shared__ unsigned s_idx [BS];

    float4 p_accel{0.f}, normal{0.f};

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
                s_pos [tid] = positions[j];
                s_pres[tid] = pressures[j];
                s_den [tid] = densities[j];
                s_idx [tid] = j;
            }
            __syncthreads();

            // Each home thread consumes the cell.
            if (active) {
                for (unsigned k = 0; k < tile_n; ++k) {
                    if (i == s_idx[k]) continue;
                    float4 r = xi - s_pos[k];
                    float r2 = dot(r, r);
                    if (r2 > SUPPORT_RADIUS * SUPPORT_RADIUS) continue;

                    float q = r_to_q(r, SUPPORT_RADIUS);
                    float4 grad_W = spiky_grad(q, SUPPORT_RADIUS) * r;
                    float dj = s_den[k];
                    float dpj = s_pres[k] / (dj * dj);
                    p_accel -= (dpi + dpj) * grad_W;

                    normal += r * cubic_spline_grad(q, SUPPORT_RADIUS) / dj;
                }
            }
            __syncthreads();
        }
    }

    if (active)
        normals[i] = SUPPORT_RADIUS * PARTICLE_MASS * normal;
    return PARTICLE_MASS * p_accel;
#else
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return {0.f};

    float4 p_accel{0.f};
    float4 xi = positions[i];
    float di = densities[i];
    float dpi = pressures[i] / (di * di);

    float4 normal{0.f};

    dev_n_search->for_neighbors<compute_pressure_accel_n_normal_u_n>(xi, [=, &p_accel, &normal] (unsigned j) {
        float4 xj = positions[j];

        if (!is_neighbor(xi, xj, i, j))
            return;

        float4 r = xi - xj;
        float q = r_to_q(r, SUPPORT_RADIUS);
        float4 grad_W = spiky_grad(q, SUPPORT_RADIUS) * r;
        float dj = densities[j];
        float dpj = pressures[j] / (dj * dj);
        p_accel -= (dpi + dpj) * grad_W;

        normal += r * cubic_spline_grad(q, SUPPORT_RADIUS) / dj;
    });

    normals[i] = SUPPORT_RADIUS * PARTICLE_MASS * normal;
    return PARTICLE_MASS * p_accel;
#endif
#endif
}

__global__ __launch_bounds__(compute_pressure_accel_n_normal_block)
void compute_pressure_accel_n_normal(
    const float4* positions, const float* densities,
    const float* pressures, float4* accelerations, float4* normals,
    unsigned n, const NSearch *dev_n_search,
    const unsigned* nl_offsets,
    const unsigned* nl_counts,
    const unsigned* nl_neighbors
) {
#if compute_pressure_accel_n_normal_shared && !compute_pressure_accel_n_normal_nlist
    const unsigned t_i = blockIdx.x;

    if (dev_n_search->table[t_i] == NSearch::EMPTY_HASH) return;
    const unsigned home_start = dev_n_search->cell_start[t_i];
    const unsigned home_end   = dev_n_search->cell_end  [t_i];
    const unsigned home_count = home_end - home_start;
    if (home_start >= home_end) return;

    constexpr unsigned BS = compute_pressure_accel_n_normal_block;
    const unsigned tiles = (home_count + BS - 1) / BS;
    for (unsigned t = 0; t < tiles; ++t) {
        float4 p_accel = compute_pressure_accel_n_normal_impl(
            positions, densities, pressures, normals,
            n, dev_n_search, nl_offsets, nl_counts, nl_neighbors, t);

        const unsigned tid = BS * t + threadIdx.x;
        const bool active = tid < home_count;
        if (!active)
            return;

        const unsigned i = home_start + tid;
        accelerations[i] = p_accel;
    }
#else
    float4 p_accel = compute_pressure_accel_n_normal_impl(
        positions, densities, pressures, normals,
        n, dev_n_search, nl_offsets, nl_counts, nl_neighbors);

    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        accelerations[i] = p_accel;
#endif
}

__global__ __launch_bounds__(compute_pressure_accel_n_normal_block)
void compute_pressure_accel_n_normal_with_boundary(
    const float4* positions, const float* densities,
    const float* pressures, float4* accelerations, float4* normals,
    unsigned fluid_n, const NSearch *fluid_n_search,
    const unsigned* nl_offsets,
    const unsigned* nl_counts,
    const unsigned* nl_neighbors,
    const float* boundary_mass, const NSearch *boundary_n_search
) {
#if compute_pressure_accel_n_normal_shared && !compute_pressure_accel_n_normal_nlist
    const unsigned t_i = blockIdx.x;

    if (fluid_n_search->table[t_i] == NSearch::EMPTY_HASH) return;
    const unsigned home_start = fluid_n_search->cell_start[t_i];
    const unsigned home_end   = fluid_n_search->cell_end  [t_i];
    const unsigned home_count = home_end - home_start;
    if (home_start >= home_end) return;

    constexpr unsigned BS = compute_pressure_accel_n_normal_block;
    const unsigned tiles = (home_count + BS - 1) / BS;
    for (unsigned t = 0; t < tiles; ++t) {
        float4 p_accel = compute_pressure_accel_n_normal_impl(
            positions, densities, pressures, normals,
            fluid_n, fluid_n_search, nl_offsets, nl_counts, nl_neighbors, t);

        const unsigned tid = BS * t + threadIdx.x;
        const bool active = tid < home_count;
        if (!active)
            return;

        const unsigned i = home_start + tid;

        float4 xi = positions[i];
        float di = densities[i];
        float pi = pressures[i];
        float dpi = pi / (di * di);
        float b_p_term = dpi + pi / (REST_DENSITY * REST_DENSITY);

        boundary_n_search->for_boundary_neighbors(xi, [=, &p_accel] (unsigned j) {
            j += fluid_n;
            float4 xj = positions[j];
            if (!is_neighbor(xi, xj, i, j))
                return;

            float4 r = xi - xj;
            float q = r_to_q(r, SUPPORT_RADIUS);
            float4 grad_W = spiky_grad(q, SUPPORT_RADIUS) * r;
            p_accel -= get_mass(boundary_mass, j, fluid_n) * b_p_term * grad_W;
        });

        accelerations[i] = p_accel;
    }
#else
    float4 p_accel = compute_pressure_accel_n_normal_impl(
        positions, densities, pressures, normals,
        fluid_n, fluid_n_search, nl_offsets, nl_counts, nl_neighbors);

    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= fluid_n)
        return;

    float4 xi = positions[i];
    float di = densities[i];
    float pi = pressures[i];
    float dpi = pi / (di * di);
    float b_p_term = dpi + pi / (REST_DENSITY * REST_DENSITY);

    boundary_n_search->for_boundary_neighbors(xi, [=, &p_accel] (unsigned j) {
        j += fluid_n;
        float4 xj = positions[j];
        if (!is_neighbor(xi, xj, i, j))
            return;

        float4 r = xi - xj;
        float q = r_to_q(r, SUPPORT_RADIUS);
        float4 grad_W = spiky_grad(q, SUPPORT_RADIUS) * r;
        p_accel -= get_mass(boundary_mass, j, fluid_n) * b_p_term * grad_W;
    });

    accelerations[i] = p_accel;
#endif
}
