#pragma once


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


__device__ inline bool is_neighbor(float4 xi, float4 xj, unsigned i, unsigned j) {
    float4 r = xi - xj;
    return i != j && dot(r, r) <= SUPPORT_RADIUS * SUPPORT_RADIUS;
}

struct NSearch {
    using cell_t = uint3;
    using hash_t = unsigned long long;
    static constexpr hash_t EMPTY_HASH = -1;
    static constexpr unsigned EMPTY_IDX = -1;
    static constexpr unsigned MAX_PARTICLES_IN_CELL = 512;
    static constexpr unsigned TABLE_SIZE = 4096;

    // https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf (eq. 34)
    __device__ static hash_t cell_hash(cell_t c) {
        return 73856093*c.x ^ 19349663*c.y ^ 83492791*c.z;
    }

    __device__ static hash_t pos_to_cell_hash(float4 pos, float cell_size) {
        return cell_hash(cell_coord(pos, cell_size));
    }

    __device__ static cell_t cell_coord(float4 pos, float cell_size) {
        return {
            signed_to_unsigned(static_cast<int>(floorf(pos.x / cell_size))),
            signed_to_unsigned(static_cast<int>(floorf(pos.y / cell_size))),
            signed_to_unsigned(static_cast<int>(floorf(pos.z / cell_size)))
        };
    }

    __device__ void insert_p_idx(const unsigned p_idx, const unsigned table_i) {
        unsigned i = 0;
        unsigned old_idx;
        do {
            old_idx = atomicCAS(&particle_indices[table_i * MAX_PARTICLES_IN_CELL + i], EMPTY_IDX, p_idx);
            ++i;
        } while (old_idx != EMPTY_IDX && i < MAX_PARTICLES_IN_CELL);

        // TODO: binary search
    }

    __device__ void insert(float4 pos, unsigned p_idx) {
        hash_t h = pos_to_cell_hash(pos, cell_size);
        unsigned i = h % TABLE_SIZE;

        hash_t old_h = atomicCAS(&table[i], EMPTY_HASH, h);

        while (old_h != EMPTY_HASH && old_h != h) {
            i = (i + 1) % TABLE_SIZE;
            old_h = atomicCAS(&table[i], EMPTY_HASH, h);
        }

        insert_p_idx(p_idx, i);
    }

    __device__ unsigned* indices_in_cell(cell_t cell) const {
        hash_t h = cell_hash(cell);
        for (unsigned j = 0; j < TABLE_SIZE; ++j) {
            unsigned t_i = (h + j) % TABLE_SIZE;
            if (table[t_i] == EMPTY_HASH)
                return nullptr;

            if (table[t_i] == h)
                return &particle_indices[MAX_PARTICLES_IN_CELL * t_i];
        }
        // Can't get here
        return nullptr;
    }

    __device__ unsigned* indices_in_cell(float4 pos) const {
        return indices_in_cell(cell_coord(pos, cell_size));
    }

    // Expects buffer to have size >= 27 * MAX_PARTICLES_IN_CELL
    __device__ unsigned list_neighbors(float4 pos, unsigned *buffer) const {
        unsigned i = 0;
        unsigned *i_ptr = &i;
        return for_neighbors(pos, [=] (unsigned p_j) {
            buffer[*i_ptr] = p_j;
            ++*i_ptr;
        });
    }

    // For neighbors with real neighbor check
    template<typename F>
    __device__ unsigned for_neighbors(const float *positions, unsigned p_i, float support_radius, F f) const {
        float4 xi = get_pos(positions, p_i);
        return for_neighbors(xi, [=] (unsigned p_j) -> void {
            if (p_i == p_j)
                return;

            float4 r = xi - get_pos(positions, p_j);
            if (dot(r, r) <= support_radius * support_radius)
                f(p_j);
        });
    }

    template<typename F>
    __device__ unsigned for_neighbors(float4 pos, const F& f) const {
        cell_t cell = cell_coord(pos, cell_size);

        unsigned i = 0;
        #pragma unroll
        for (int x = -1; x <= 1; ++x) {
            #pragma unroll
            for (int y = -1; y <= 1; ++y) {
                #pragma unroll
                for (int z = -1; z <= 1; ++z) {
                    cell_t n_cell{cell.x + x, cell.y + y, cell.z + z};
                    unsigned *indices_start = indices_in_cell(n_cell);

                    if (indices_start == nullptr)
                        continue;

                    for (unsigned j = 0; j < MAX_PARTICLES_IN_CELL; ++j) {
                        if (indices_start[j] == EMPTY_IDX)
                            break;

                        f(indices_start[j]);
                        ++i;
                    }
                }
            }
        }
        return i;
    }

    // Stores hashes of cells or empty
    // - linear probing
    // - index in table == index of array in particle_indices
    hash_t *table;

    // 2D arrays of particle indices into particle_positions
    unsigned *particle_indices;

    float cell_size;
};
