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
    static constexpr hash_t EMPTY_HASH = -1;
    static constexpr unsigned EMPTY_CELL = -1;
    // static constexpr unsigned TABLE_SIZE = 129536;

    // https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf (eq. 34)
    __device__ static hash_t cell_hash(cell_t c) {
        return 73856093*c.x ^ 19349663*c.y ^ 83492791*c.z;
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
    __device__ __host__ unsigned find_cell_in_table(cell_t cell) const {
        hash_t h = cell_hash(cell);
        for (unsigned j = 0; j < T_SIZE; ++j) {
            unsigned t_i = (h + j) % T_SIZE;
            if (table[t_i] == EMPTY_HASH)
                return EMPTY_CELL;
            if (table[t_i] == h)
                return t_i;
        }
        // Can't get here
        return EMPTY_CELL;
    }

    __device__ unsigned add_cell(hash_t h) {
        unsigned i = h % TABLE_SIZE;
        for (unsigned j = 0; j < TABLE_SIZE; ++j) {
            hash_t old_h = atomicCAS(&table[i], EMPTY_HASH, h);
            if (old_h == EMPTY_HASH || old_h == h)
                return i;
            i = (i + 1) % TABLE_SIZE;
        }
        // Can't get here
        return -1;
    }

    __device__ void set_cell_start(hash_t h, unsigned start) {
        unsigned cell_i = add_cell(h);
        cell_start[cell_i] = start;
    }

    __device__ void set_cell_end(hash_t h, unsigned end) {
        unsigned cell_i = add_cell(h);
        cell_end[cell_i] = end;
    }

    template<int U_N, typename F>
    __device__ __host__ unsigned for_neighbors(float4 pos, F f) const {
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
    __device__ __host__ unsigned for_boundary_neighbors(float4 pos, F f) const {
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
