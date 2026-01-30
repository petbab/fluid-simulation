#pragma once

#include "nsearch.cuh"

class NSearchWrapper {
public:
    explicit NSearchWrapper(float cell_size) {
        dev_n_search = new_n_search(host_n_search, cell_size);
    }

    ~NSearchWrapper() {
        delete_n_search(dev_n_search, host_n_search);
    }

    void rebuild(const float *particle_positions, unsigned n) {
        rebuild_n_search(dev_n_search, host_n_search, particle_positions, n);
    }

    const NSearch* dev_ptr() const {
        return dev_n_search;
    }

private:
    NSearch *dev_n_search;
    NSearch host_n_search;
};

struct NSearchHost {
    static NSearchHost copy_from_device(const NSearch *dev_n_search) {
        NSearch h_n_search;
        cudaMemcpy(&h_n_search, dev_n_search, sizeof(NSearch), cudaMemcpyDeviceToHost);
        cudaCheckError();

        NSearchHost n_search{};
        cudaMemcpy(n_search.table.data(), h_n_search.table,
        sizeof(NSearch::hash_t) * NSearch::TABLE_SIZE, cudaMemcpyDeviceToHost);
        cudaCheckError();
        cudaMemcpy(n_search.particle_indices.data(), h_n_search.particle_indices,
            sizeof(unsigned) * NSearch::TABLE_SIZE * NSearch::MAX_PARTICLES_IN_CELL, cudaMemcpyDeviceToHost);
        cudaCheckError();
        n_search.cell_size = h_n_search.cell_size;

        return n_search;
    }

    std::array<NSearch::hash_t, NSearch::TABLE_SIZE> table;
    std::array<unsigned, NSearch::TABLE_SIZE * NSearch::MAX_PARTICLES_IN_CELL> particle_indices;
    float cell_size;
};
