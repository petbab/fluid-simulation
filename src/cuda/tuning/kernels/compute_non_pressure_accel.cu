#ifdef NOT_IN_KTT
#include "common.cuh"
#endif

#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)

#ifndef EXTERNAL_FORCE
#define EXTERNAL_FORCE ([](float4 pos) { return make_float4(0., 0., 0., 0.); })
#endif
#ifndef compute_non_pressure_accel_u_n
#define compute_non_pressure_accel_u_n 1
#endif
#ifndef compute_non_pressure_accel_block
#define compute_non_pressure_accel_block 128
#endif
#ifndef compute_non_pressure_accel_cache_log2
#define compute_non_pressure_accel_cache_log2 2  // CACHE = BS << 2
#endif


static constexpr float4 GRAVITY{0, -9.81f, 0, 0};
static constexpr float VISCOSITY = 0.001f;
static constexpr float SURFACE_TENSION_ALPHA = 0.13f;


__global__ __launch_bounds__(compute_non_pressure_accel_block)
void compute_non_pressure_accel(
    const float4* positions, const float* densities,
    const float4* velocities, const float4* normals, float4* acceleration,
    unsigned n, const NSearch *dev_n_search,
    const unsigned* nl_offsets,
    const unsigned* nl_counts,
    const unsigned* nl_neighbors
) {
#if compute_non_pressure_accel_nlist
#if compute_non_pressure_accel_shared
    constexpr unsigned BS    = compute_non_pressure_accel_block;
    constexpr unsigned CACHE = BS << (2 + compute_non_pressure_accel_cache_log2);
    constexpr unsigned MASK  = CACHE - 1;
    constexpr unsigned EMPTY = ~0u;

    __shared__ unsigned s_keys[CACHE];
    __shared__ float4   s_xj  [CACHE];
    __shared__ float4   s_vj  [CACHE];
    __shared__ float4   s_nj  [CACHE];
    __shared__ float    s_dj  [CACHE];

    const unsigned tid    = threadIdx.x;
    const unsigned i      = blockIdx.x * BS + tid;
    const bool     active = i < n;
    const unsigned i_safe = active ? i : 0u;

    const float4 xi = positions [i_safe];
    const float4 vi = velocities[i_safe];
    const float4 ni = normals   [i_safe];
    const float  di = densities [i_safe];

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
                    s_vj[slot] = velocities[j];
                    s_nj[slot] = normals   [j];
                    s_dj[slot] = densities [j];
                    break;
                }
                if (old == j) break;
            }
            slot = (slot + 1) & MASK;
        }
    }
    __syncthreads();

    // Shared memory only consumption.
    float4 f{0.f}, vlap{0.f};
    for (unsigned k = 0; k < count; ++k) {
        const unsigned j = nl_neighbors[start + k];
        unsigned slot = hash(j);
        for (int c = 0; s_keys[slot] != j && c < CACHE; ++c) slot = (slot + 1) & MASK;

        const float4 xj = s_xj[slot];
        const float4 vj = s_vj[slot];
        const float4 nj = s_nj[slot];
        const float  dj = s_dj[slot];

        const float4 x_ij = xi - xj;
        const float  r2   = dot(x_ij, x_ij);
        const float  q    = r_to_q(x_ij, SUPPORT_RADIUS);

        if (r2 > 1e-6f)
            f += (PARTICLE_MASS * normalize(x_ij) * cohesion(q, SUPPORT_RADIUS)
                  + ni - nj) / (di + dj);

        const float4 v_ij = vi - vj;
        vlap += dot(v_ij, x_ij) * cubic_spline_grad(q, SUPPORT_RADIUS) * x_ij
              / (dj * (r2 + 0.01f * SUPPORT_RADIUS * SUPPORT_RADIUS));
    }

    if (active)
        acceleration[i] = GRAVITY + EXTERNAL_FORCE(xi)
            + VISCOSITY * 10.f * PARTICLE_MASS * vlap
            - SURFACE_TENSION_ALPHA * 2.f * REST_DENSITY * f;

#else
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 xi = positions[i];
    float4 vi = velocities[i];
    float4 ni = normals[i];
    float  di = densities[i];
    float4 f{0.f}, vlap{0.f};

    const unsigned start = nl_offsets[i];
    const unsigned end   = start + nl_counts[i];
    #pragma unroll compute_non_pressure_accel_u_n
    for (unsigned k = start; k < end; ++k) {
        unsigned j = nl_neighbors[k];
        float4 x_ij = xi - positions[j];
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

    acceleration[i] = GRAVITY + EXTERNAL_FORCE(xi)
        + VISCOSITY * 10.f * PARTICLE_MASS * vlap
        - SURFACE_TENSION_ALPHA * 2.f * REST_DENSITY * f;
#endif
#else
#if compute_non_pressure_accel_shared
    constexpr unsigned BS = compute_non_pressure_accel_block;

    const unsigned t_i = blockIdx.x;
    const unsigned tid = threadIdx.x;

    if (dev_n_search->table[t_i] == NSearch::EMPTY_HASH) return;
    const unsigned home_start = dev_n_search->cell_start[t_i];
    const unsigned home_end   = dev_n_search->cell_end  [t_i];
    const unsigned home_count = home_end - home_start;
    if (home_start >= home_end) return;

    const bool active = tid < home_count;
    const unsigned i = active ? home_start + tid : home_start;

    const float4 xi = positions [i];
    const float4 vi = velocities[i];
    const float4 ni = normals   [i];
    const float  di = densities [i];

    const NSearch::cell_t anchor = NSearch::cell_coord(xi);

    __shared__ float4   s_pos [BS];
    __shared__ float4   s_vel [BS];
    __shared__ float4   s_norm[BS];
    __shared__ float    s_den [BS];
    __shared__ unsigned s_idx [BS];

    float4 f{0.f}, vlap{0.f};

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
                s_pos [tid] = positions [j];
                s_vel [tid] = velocities[j];
                s_norm[tid] = normals   [j];
                s_den [tid] = densities [j];
                s_idx [tid] = j;
            }
            __syncthreads();

            // Each home thread consumes the cell.
            if (active) {
                for (unsigned k = 0; k < tile_n; ++k) {
                    if (i == s_idx[k]) continue;
                    float4 x_ij = xi - s_pos[k];
                    float r2 = dot(x_ij, x_ij);
                    if (r2 > SUPPORT_RADIUS * SUPPORT_RADIUS) continue;

                    float dj = s_den[k];
                    float q = r_to_q(x_ij, SUPPORT_RADIUS);
                    float4 nj = s_norm[k];

                    if (r2 > 1e-6f)
                        f += (PARTICLE_MASS * normalize(x_ij) * cohesion(q, SUPPORT_RADIUS)
                              + ni - nj) / (di + dj);

                    float4 v_ij = vi - s_vel[k];
                    vlap += dot(v_ij, x_ij) * cubic_spline_grad(q, SUPPORT_RADIUS) * x_ij
                          / (dj * (r2 + 0.01f * SUPPORT_RADIUS * SUPPORT_RADIUS));
                }
            }
            __syncthreads();
        }
    }

    if (active) {
        acceleration[i] = GRAVITY + EXTERNAL_FORCE(xi)
            + VISCOSITY * 10.f * PARTICLE_MASS * vlap
            - SURFACE_TENSION_ALPHA * 2.f * REST_DENSITY * f;
    }
#else
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 xi = positions[i];
    float4 vi = velocities[i];
    float4 ni = normals[i];
    float  di = densities[i];
    float4 f{0.f}, vlap{0.f};

    dev_n_search->for_neighbors<compute_non_pressure_accel_u_n>(xi, [=, &f, &vlap] (unsigned j) {
        float4 xj = positions[j];
        if (!is_neighbor(xi, xj, i, j))
            return;

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
    });

    acceleration[i] = GRAVITY + EXTERNAL_FORCE(xi)
        + VISCOSITY * 10.f * PARTICLE_MASS * vlap
        - SURFACE_TENSION_ALPHA * 2.f * REST_DENSITY * f;
#endif
#endif
}
