#pragma once

#ifdef NOT_IN_KTT
#include "cuda/math.cuh"
#include "cuda/util.cuh"
#include "cuda/SPH/kernel.cuh"
#include "cuda/nsearch/morton.cuh"
#endif

#define CUDA_DIR KERNEL_DIR/../..
#define CUDA_PATH(file) <CUDA_DIR ## file>

#include CUDA_PATH(/math.cuh)
#include CUDA_PATH(/util.cuh)
#include CUDA_PATH(/SPH/kernel.cuh)
#include CUDA_PATH(/nsearch/morton.cuh)

static constexpr float PARTICLE_RADIUS = 0.02f;
static constexpr float PARTICLE_SPACING = 2.f * PARTICLE_RADIUS;
static constexpr float SUPPORT_RADIUS = 2.f * PARTICLE_SPACING;
static constexpr float REST_DENSITY = 1000.f;
static constexpr float PARTICLE_VOLUME = PARTICLE_SPACING * PARTICLE_SPACING * PARTICLE_SPACING * 0.8f;
static constexpr float PARTICLE_MASS = REST_DENSITY * PARTICLE_VOLUME;

#ifndef BOUNDARY_TABLE_SIZE
#define BOUNDARY_TABLE_SIZE TABLE_SIZE
#endif

#ifndef CELL_SIZE_MULT
#define CELL_SIZE_MULT 1.f
#endif

static constexpr float CELL_SIZE = SUPPORT_RADIUS * CELL_SIZE_MULT;

__device__ inline bool is_neighbor(float4 xi, float4 xj, unsigned i, unsigned j) {
    float4 r = xi - xj;
    return i != j && dot(r, r) <= SUPPORT_RADIUS * SUPPORT_RADIUS;
}

__device__ inline bool is_neighbor(float4 xi, float4 xj) {
    float4 r = xi - xj;
    return dot(r, r) <= SUPPORT_RADIUS * SUPPORT_RADIUS;
}

struct NSearch {
    using cell_t = uint3;
    using hash_t = unsigned long long;
    static constexpr hash_t EMPTY_HASH = ~0ULL;
    static constexpr unsigned EMPTY_CELL = ~0u;

    __device__ static hash_t pack(cell_t c) {
        return static_cast<hash_t>(c.x) | (static_cast<hash_t>(c.y) << 21) | (static_cast<hash_t>(c.z) << 42);
    }
    __device__ static cell_t unpack(hash_t p) {
        return { (unsigned)(p & 0x1FFFFFu),
                 (unsigned)((p >> 21) & 0x1FFFFFu),
                 (unsigned)((p >> 42) & 0x1FFFFFu) };
    }

    // https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf (eq. 34)
    __device__ static hash_t cell_hash(cell_t c) {
        return 73856093ull*c.x ^ 19349663ull*c.y ^ 83492791ull*c.z;
    }

    __device__ static hash_t pos_to_cell_hash(float4 pos) {
        return cell_hash(cell_coord(pos));
    }

    __device__ static cell_t cell_coord(float4 pos) {
        return {
            signed_to_unsigned(static_cast<int>(floorf(pos.x / CELL_SIZE))),
            signed_to_unsigned(static_cast<int>(floorf(pos.y / CELL_SIZE))),
            signed_to_unsigned(static_cast<int>(floorf(pos.z / CELL_SIZE)))
        };
    }

    __device__ static cell_t cell_coord_boundary(float4 pos) {
        return {
            signed_to_unsigned(static_cast<int>(floorf(pos.x / SUPPORT_RADIUS))),
            signed_to_unsigned(static_cast<int>(floorf(pos.y / SUPPORT_RADIUS))),
            signed_to_unsigned(static_cast<int>(floorf(pos.z / SUPPORT_RADIUS)))
        };
    }

    template<unsigned T_SIZE>
    __device__ unsigned find_cell_in_table(cell_t c) const {
        hash_t pk = pack(c);
        unsigned t_i = cell_hash(c) % T_SIZE;
        for (unsigned j = 0; j < T_SIZE; ++j) {
            hash_t k = table[t_i];
            if (k == EMPTY_HASH)
                return EMPTY_CELL;
            if (k == pk)
                return t_i;
            t_i = (t_i + 1) % T_SIZE;
        }
        return EMPTY_CELL;
    }

    __device__ unsigned add_cell(cell_t c) {
        hash_t pk = pack(c);
        unsigned i = cell_hash(c) % TABLE_SIZE;
        for (unsigned j = 0; j < TABLE_SIZE; ++j) {
            hash_t old = atomicCAS((unsigned long long*)&table[i], EMPTY_HASH, pk);
            if (old == EMPTY_HASH || old == pk)
                return i;
            i = (i + 1) % TABLE_SIZE;
        }
        return -1;
    }

    __device__ void set_cell_start(cell_t c, unsigned start) {
        unsigned cell_i = add_cell(c);
        cell_start[cell_i] = start;
    }

    __device__ void set_cell_end(cell_t c, unsigned end) {
        unsigned cell_i = add_cell(c);
        cell_end[cell_i] = end;
    }

    template<int U_N, typename F>
    __device__ unsigned for_neighbors(float4 pos, F f) const {
        cell_t cell = cell_coord(pos);

        unsigned count = 0;
        static constexpr int RANGE = static_cast<int>(SUPPORT_RADIUS / CELL_SIZE + .5f);
        for (int x = -RANGE; x <= RANGE; ++x) {
            for (int y = -RANGE; y <= RANGE; ++y) {
                #pragma unroll U_N
                for (int z = -RANGE; z <= RANGE; ++z) {
                    cell_t n_cell{cell.x + x, cell.y + y, cell.z + z};
                    unsigned t_i = find_cell_in_table<TABLE_SIZE>(n_cell);
                    if (t_i == EMPTY_CELL)
                        continue;

                    const unsigned start = cell_start[t_i];
                    const unsigned end = cell_end[t_i];
                    count += end - start;

                    for (unsigned j = start; j < end; ++j)
                        f(j);
                }
            }
        }
        return count;
    }

    template<typename F>
    __device__ unsigned for_boundary_neighbors(float4 pos, F f) const {
        cell_t cell = cell_coord_boundary(pos);

        unsigned count = 0;
        static constexpr int RANGE = 1;
        for (int x = -RANGE; x <= RANGE; ++x) {
            for (int y = -RANGE; y <= RANGE; ++y) {
                // #pragma unroll
                for (int z = -RANGE; z <= RANGE; ++z) {
                    cell_t n_cell{cell.x + x, cell.y + y, cell.z + z};
                    unsigned t_i = find_cell_in_table<BOUNDARY_TABLE_SIZE>(n_cell);
                    if (t_i == EMPTY_CELL)
                        continue;

                    const unsigned start = cell_start[t_i];
                    const unsigned end = cell_end[t_i];
                    count += end - start;

                    for (unsigned j = start; j < end; ++j)
                        f(j);
                }
            }
        }
        return count;
    }

    // Stores hashes of cells or empty
    // - linear probing
    // - index in table == index in cell_start/end
    hash_t *table;
    unsigned table_size;

    // Indices into sorted particle indices,
    // cell_end is non-inclusive
    unsigned *cell_start, *cell_end;

    float cell_size;
};
