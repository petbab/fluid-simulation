#pragma once

#include <cuda/util.cuh>
#include <debug.h>


/**
 * @brief GPU spatial hash table for neighbor search.
 *
 * Uses linear probing with cell hashes. The table stores hashes of occupied cells;
 * cell_start and cell_end index into a sorted particle array.
 */
struct NSearch {
    using cell_t = uint3;                ///< Cell coordinate type.
    using hash_t = unsigned long long;   ///< Hash value type.
    static constexpr hash_t EMPTY_HASH = std::numeric_limits<hash_t>::max();  ///< Sentinel for empty hash.
    static constexpr unsigned EMPTY_CELL = std::numeric_limits<unsigned>::max();  ///< Sentinel for empty cell.

    /**
     * @brief Hash table storing cell hashes (linear probing).
     *
     * Index in table == index in cell_start/cell_end.
     */
    hash_t *table;
    unsigned table_size;  ///< Number of slots in the hash table.

    /**
     * @brief Start/end indices into sorted particle indices.
     *
     * cell_end is non-inclusive.
     */
    unsigned *cell_start, *cell_end;

    float cell_size;  ///< Spatial cell size.
};

/**
 * @brief Clears the neighbor search hash table on the device.
 * @param host_n_search Host-side NSearch descriptor with device pointers.
 */
__host__ inline void clear_n_search(const NSearch& host_n_search) {
    cudaMemset(host_n_search.table, 0xff, sizeof(NSearch::hash_t) * host_n_search.table_size); cudaCheckError();
    cudaMemset(host_n_search.cell_start, 0, sizeof(unsigned) * host_n_search.table_size); cudaCheckError();
    cudaMemset(host_n_search.cell_end, 0, sizeof(unsigned) * host_n_search.table_size); cudaCheckError();
}

/**
 * @brief Allocates and initializes a device NSearch structure.
 * @param host_n_search Host NSearch to populate with device pointers.
 * @param table_size Number of hash table slots.
 * @param cell_size Spatial cell size.
 * @return Device pointer to the allocated NSearch.
 */
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

/**
 * @brief Frees all device memory associated with an NSearch.
 * @param dev_n_search Device pointer returned by new_n_search().
 * @param host_n_search Host NSearch descriptor with device pointers.
 */
__host__ inline void delete_n_search(NSearch* dev_n_search, const NSearch& host_n_search) {
    cudaFree(host_n_search.table); cudaCheckError();
    cudaFree(host_n_search.cell_start); cudaCheckError();
    cudaFree(host_n_search.cell_end); cudaCheckError();
    cudaFree(dev_n_search); cudaCheckError();
}

/**
 * @brief Shallow-copies host NSearch data to the device structure.
 * @param dev_n_search Device NSearch pointer.
 * @param host_n_search Host NSearch with updated fields.
 */
__host__ inline void shallow_copy_n_search(NSearch *dev_n_search, const NSearch &host_n_search) {
    cudaMemcpy(dev_n_search, &host_n_search, sizeof(NSearch), cudaMemcpyHostToDevice);
    cudaCheckError();
}

/**
 * @brief Host-side copy of NSearch data for debugging and statistics.
 */
struct NSearchHost {
    /**
     * @brief Copies NSearch data from device to host.
     * @param dev_n_search Device NSearch pointer.
     * @return Host copy of the neighbor search structure.
     */
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

    std::vector<NSearch::hash_t> table;  ///< Copied hash table.
    unsigned table_size = 0;             ///< Table size.
    std::vector<unsigned> cell_start;    ///< Copied cell start indices.
    std::vector<unsigned> cell_end;      ///< Copied cell end indices.
    float cell_size = 0.f;               ///< Cell size.
};
