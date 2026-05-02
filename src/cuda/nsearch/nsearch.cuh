#pragma once

#include <cuda/util.cuh>
#include <debug.h>
#include "morton.cuh"


struct NSearch {
    using cell_t = uint3;
    using hash_t = unsigned long long;
    static constexpr hash_t EMPTY_HASH = std::numeric_limits<hash_t>::max();
    static constexpr unsigned EMPTY_CELL = std::numeric_limits<unsigned>::max();
    static constexpr unsigned TABLE_SIZE = 16192;

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

    __device__ __host__ unsigned find_cell_in_table(cell_t cell) const {
        hash_t h = cell_hash(cell);
        for (unsigned j = 0; j < TABLE_SIZE; ++j) {
            unsigned t_i = (h + j) % TABLE_SIZE;
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
        return std::numeric_limits<unsigned>::max();
    }

    __device__ void set_cell_start(hash_t h, unsigned start) {
        unsigned cell_i = add_cell(h);
        cell_start[cell_i] = start;
    }

    __device__ void set_cell_end(hash_t h, unsigned end) {
        unsigned cell_i = add_cell(h);
        cell_end[cell_i] = end;
    }

    template<typename F>
    __device__ __host__ unsigned for_neighbors(float4 pos, F f) const {
        cell_t cell = cell_coord(pos, cell_size);
        unsigned count = 0;
        #pragma unroll
        for (int x = -1; x <= 1; ++x) {
            #pragma unroll
            for (int y = -1; y <= 1; ++y) {
                #pragma unroll
                for (int z = -1; z <= 1; ++z) {
                    cell_t n_cell{cell.x + x, cell.y + y, cell.z + z};
                    unsigned t_i = find_cell_in_table(n_cell);
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

    // Indices into sorted particle indices,
    // cell_end is non-inclusive
    unsigned *cell_start, *cell_end;

    float cell_size;
};

__host__ inline void clear_n_search(const NSearch& host_n_search) {
    cudaMemset(host_n_search.table, 0xff, sizeof(NSearch::hash_t) * NSearch::TABLE_SIZE); cudaCheckError();
    cudaMemset(host_n_search.cell_start, 0, sizeof(unsigned) * NSearch::TABLE_SIZE); cudaCheckError();
    cudaMemset(host_n_search.cell_end, 0, sizeof(unsigned) * NSearch::TABLE_SIZE); cudaCheckError();
}

__host__ inline NSearch* new_n_search(NSearch &host_n_search, float cell_size) {
    NSearch* dev_n_search;
    cudaMalloc(&dev_n_search, sizeof(NSearch)); cudaCheckError();

    cudaMalloc(&host_n_search.table, sizeof(NSearch::hash_t) * NSearch::TABLE_SIZE); cudaCheckError();
    cudaMalloc(&host_n_search.cell_start, sizeof(unsigned) * NSearch::TABLE_SIZE); cudaCheckError();
    cudaMalloc(&host_n_search.cell_end, sizeof(unsigned) * NSearch::TABLE_SIZE); cudaCheckError();
    host_n_search.cell_size = cell_size;

    cudaMemcpy(dev_n_search, &host_n_search, sizeof(NSearch), cudaMemcpyHostToDevice); cudaCheckError();

    clear_n_search(host_n_search);

    return dev_n_search;
}

__host__ inline void delete_n_search(NSearch* dev_n_search, const NSearch& host_n_search) {
    cudaFree(host_n_search.table); cudaCheckError();
    cudaFree(host_n_search.cell_start); cudaCheckError();
    cudaFree(host_n_search.cell_end); cudaCheckError();
    cudaFree(dev_n_search); cudaCheckError();
}

struct NSearchHost {
    static NSearchHost copy_from_device(const NSearch *dev_n_search) {
        NSearch h_n_search;
        cudaMemcpy(&h_n_search, dev_n_search, sizeof(NSearch), cudaMemcpyDeviceToHost);
        cudaCheckError();

        NSearchHost n_search;
        n_search.table.resize(NSearch::TABLE_SIZE);
        n_search.cell_start.resize(NSearch::TABLE_SIZE);
        n_search.cell_end.resize(NSearch::TABLE_SIZE);
        n_search.cell_size = h_n_search.cell_size;

        cudaMemcpy(n_search.table.data(), h_n_search.table,
        sizeof(NSearch::hash_t) * NSearch::TABLE_SIZE, cudaMemcpyDeviceToHost);
        cudaCheckError();
        cudaMemcpy(n_search.cell_start.data(), h_n_search.cell_start,
            sizeof(unsigned) * NSearch::TABLE_SIZE, cudaMemcpyDeviceToHost);
        cudaCheckError();
        cudaMemcpy(n_search.cell_end.data(), h_n_search.cell_end,
            sizeof(unsigned) * NSearch::TABLE_SIZE, cudaMemcpyDeviceToHost);
        cudaCheckError();

        return n_search;
    }

    std::vector<NSearch::hash_t> table;
    std::vector<unsigned> cell_start;
    std::vector<unsigned> cell_end;
    float cell_size = 0.f;
};
