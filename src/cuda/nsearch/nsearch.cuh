#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/math.cuh>
#include <debug.h>
#include "morton.cuh"


__device__ __host__ inline float4 get_pos(const float *positions, unsigned i) {
    unsigned ii = 3 * i;
    return {positions[ii], positions[ii + 1], positions[ii + 2]};
}

struct NSearch {
    using cell_t = uint3;
    using hash_t = unsigned long long;
    static constexpr hash_t EMPTY_HASH = std::numeric_limits<hash_t>::max();
    static constexpr unsigned EMPTY_IDX = std::numeric_limits<unsigned>::max();
    static constexpr unsigned MAX_PARTICLES_IN_CELL = 512;
    static constexpr unsigned TABLE_SIZE = 4096;

    // https://sph-tutorial.physics-simulation.org/pdf/SPH_Tutorial.pdf (eq. 34)
    __device__ __host__ static hash_t cell_hash(cell_t c) {
        return 73856093*c.x ^ 19349663*c.y ^ 83492791*c.z;
    }

    __device__ __host__ static hash_t pos_to_cell_hash(float4 pos, float cell_size) {
        return cell_hash(cell_coord(pos, cell_size));
    }

    __device__ __host__ static cell_t cell_coord(float4 pos, float cell_size) {
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

    __device__ __host__ unsigned* indices_in_cell(cell_t cell) const {
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

    __device__ __host__ unsigned* indices_in_cell(float4 pos) const {
        return indices_in_cell(cell_coord(pos, cell_size));
    }

    // Expects buffer to have size >= 27 * MAX_PARTICLES_IN_CELL
    __device__ __host__ unsigned list_neighbors(float4 pos, unsigned *buffer) const {
        unsigned i = 0;
        unsigned *i_ptr = &i;
        return for_neighbors(pos, [=] __device__ __host__ (unsigned p_j) {
            buffer[*i_ptr] = p_j;
            ++*i_ptr;
        });
    }

    // For neighbors with real neighbor check
    template<typename F>
    __device__ __host__ unsigned for_neighbors(const float *positions, unsigned p_i, float support_radius, F f) const {
        float4 xi = get_pos(positions, p_i);
        return for_neighbors(xi, [=] __device__ __host__ (unsigned p_j) -> void {
            if (p_i == p_j)
                return;

            float4 r = xi - get_pos(positions, p_j);
            if (dot(r, r) <= support_radius * support_radius)
                f(p_j);
        });
    }

    template<typename F>
    __device__ __host__ unsigned for_neighbors(float4 pos, F f) const {
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

__host__ inline NSearch* new_n_search(NSearch &host_n_search, float cell_size) {
    NSearch* dev_n_search;
    cudaMalloc(&dev_n_search, sizeof(NSearch)); cudaCheckError();

    cudaMalloc(&host_n_search.table, sizeof(NSearch::hash_t) * NSearch::TABLE_SIZE); cudaCheckError();
    cudaMalloc(&host_n_search.particle_indices,
        sizeof(unsigned) * NSearch::TABLE_SIZE * NSearch::MAX_PARTICLES_IN_CELL); cudaCheckError();
    host_n_search.cell_size = cell_size;

    cudaMemcpy(dev_n_search, &host_n_search, sizeof(NSearch), cudaMemcpyHostToDevice); cudaCheckError();

    cudaMemset(host_n_search.table, 0xff, sizeof(NSearch::hash_t) * NSearch::TABLE_SIZE); cudaCheckError();
    cudaMemset(host_n_search.particle_indices, 0xff,
        sizeof(unsigned) * NSearch::TABLE_SIZE * NSearch::MAX_PARTICLES_IN_CELL); cudaCheckError();

    return dev_n_search;
}

__host__ inline void clear_n_search(const NSearch& host_n_search) {
    cudaMemset(host_n_search.table, 0xff, sizeof(NSearch::hash_t) * NSearch::TABLE_SIZE); cudaCheckError();
    cudaMemset(host_n_search.particle_indices, 0xff,
        sizeof(unsigned) * NSearch::TABLE_SIZE * NSearch::MAX_PARTICLES_IN_CELL); cudaCheckError();
}

__host__ inline void delete_n_search(NSearch* dev_n_search, const NSearch& host_n_search) {
    cudaFree(host_n_search.table); cudaCheckError();
    cudaFree(host_n_search.particle_indices); cudaCheckError();
    cudaFree(dev_n_search); cudaCheckError();
}

__global__ inline void rebuild_n_search_k(NSearch *dev_n_search, const float *particle_positions, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 pos = get_pos(particle_positions, i);
    dev_n_search->insert(pos, i);
}

#define BLOCK_SIZE 128

__host__ inline void rebuild_n_search(
    NSearch *dev_n_search,
    const NSearch &host_n_search,
    const float *particle_positions,
    unsigned n
) {
    clear_n_search(host_n_search);

    rebuild_n_search_k<<<BLOCK_SIZE, n / BLOCK_SIZE + 1>>>(dev_n_search, particle_positions, n);
}

struct NSearchHost {
    static NSearchHost copy_from_device(const NSearch *dev_n_search) {
        NSearch h_n_search;
        cudaMemcpy(&h_n_search, dev_n_search, sizeof(NSearch), cudaMemcpyDeviceToHost);
        cudaCheckError();

        NSearchHost n_search;
        n_search.table.resize(NSearch::TABLE_SIZE);
        n_search.particle_indices.resize(NSearch::TABLE_SIZE * NSearch::MAX_PARTICLES_IN_CELL);
        n_search.cell_size = h_n_search.cell_size;

        cudaMemcpy(n_search.table.data(), h_n_search.table,
        sizeof(NSearch::hash_t) * NSearch::TABLE_SIZE, cudaMemcpyDeviceToHost);
        cudaCheckError();
        cudaMemcpy(n_search.particle_indices.data(), h_n_search.particle_indices,
            sizeof(unsigned) * NSearch::TABLE_SIZE * NSearch::MAX_PARTICLES_IN_CELL, cudaMemcpyDeviceToHost);
        cudaCheckError();

        return n_search;
    }

    std::vector<NSearch::hash_t> table;
    std::vector<unsigned> particle_indices;
    float cell_size = 0.f;
};
