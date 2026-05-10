#pragma once

#include <cuda/util.cuh>
#include <debug.h>


struct NSearch {
    using cell_t = uint3;
    using hash_t = unsigned long long;
    static constexpr hash_t EMPTY_HASH = std::numeric_limits<hash_t>::max();
    static constexpr unsigned EMPTY_CELL = std::numeric_limits<unsigned>::max();

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

__host__ inline void clear_n_search(const NSearch& host_n_search) {
    cudaMemset(host_n_search.table, 0xff, sizeof(NSearch::hash_t) * host_n_search.table_size); cudaCheckError();
    cudaMemset(host_n_search.cell_start, 0, sizeof(unsigned) * host_n_search.table_size); cudaCheckError();
    cudaMemset(host_n_search.cell_end, 0, sizeof(unsigned) * host_n_search.table_size); cudaCheckError();
}

__host__ inline NSearch* new_n_search(NSearch &host_n_search, unsigned table_size, float cell_size) {
    NSearch* dev_n_search;
    cudaMalloc(&dev_n_search, sizeof(NSearch)); cudaCheckError();

    host_n_search.table_size = table_size;
    cudaMalloc(&host_n_search.table, sizeof(NSearch::hash_t) * host_n_search.table_size); cudaCheckError();
    cudaMalloc(&host_n_search.cell_start, sizeof(unsigned) * host_n_search.table_size); cudaCheckError();
    cudaMalloc(&host_n_search.cell_end, sizeof(unsigned) * host_n_search.table_size); cudaCheckError();
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

__host__ inline void shallow_copy_n_search(NSearch *dev_n_search, const NSearch &host_n_search) {
    cudaMemcpy(dev_n_search, &host_n_search, sizeof(NSearch), cudaMemcpyHostToDevice);
    cudaCheckError();
}

struct NSearchHost {
    static NSearchHost copy_from_device(const NSearch *dev_n_search) {
        NSearch h_n_search;
        cudaMemcpy(&h_n_search, dev_n_search, sizeof(NSearch), cudaMemcpyDeviceToHost);
        cudaCheckError();

        NSearchHost n_search;
        n_search.table.resize(h_n_search.table_size);
        n_search.cell_start.resize(h_n_search.table_size);
        n_search.cell_end.resize(h_n_search.table_size);
        n_search.cell_size = h_n_search.cell_size;
        n_search.table_size = h_n_search.table_size;

        cudaMemcpy(n_search.table.data(), h_n_search.table,
        sizeof(NSearch::hash_t) * h_n_search.table_size, cudaMemcpyDeviceToHost);
        cudaCheckError();
        cudaMemcpy(n_search.cell_start.data(), h_n_search.cell_start,
            sizeof(unsigned) * h_n_search.table_size, cudaMemcpyDeviceToHost);
        cudaCheckError();
        cudaMemcpy(n_search.cell_end.data(), h_n_search.cell_end,
            sizeof(unsigned) * h_n_search.table_size, cudaMemcpyDeviceToHost);
        cudaCheckError();

        return n_search;
    }

    std::vector<NSearch::hash_t> table;
    unsigned table_size = 0;
    std::vector<unsigned> cell_start;
    std::vector<unsigned> cell_end;
    float cell_size = 0.f;
};
