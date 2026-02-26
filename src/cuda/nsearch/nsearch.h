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
