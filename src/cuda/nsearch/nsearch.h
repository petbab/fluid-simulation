#pragma once

#include "nsearch.cuh"
#include "cuda/tuning/rebuild_n_search.cuh"


class NSearchWrapper {
public:
    explicit NSearchWrapper(float cell_size, unsigned total_particles)
        : rebuild_n_search_tuner(total_particles), total_particles(total_particles) {
        dev_n_search = new_n_search(host_n_search, cell_size);
    }

    ~NSearchWrapper() {
        delete_n_search(dev_n_search, host_n_search);
    }

    void rebuild(float *particle_positions) {
        clear_n_search(host_n_search);
        rebuild_n_search_tuner.run(dev_n_search, particle_positions, total_particles);
    }

    const NSearch* dev_ptr() const { return dev_n_search; }
    NSearch* dev_ptr() { return dev_n_search; }

private:
    NSearch *dev_n_search;
    NSearch host_n_search;
    RebuildNSearchTuner rebuild_n_search_tuner;
    unsigned total_particles;
};
