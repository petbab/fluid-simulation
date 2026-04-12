#pragma once

#include <memory>
#include "nsearch.cuh"
#include "cuda/tuning/rebuild_n_search.cuh"


class NSearchWrapper {
public:
    NSearchWrapper(float cell_size, unsigned total_particles)
        : total_particles(total_particles) {
        dev_n_search = new_n_search(host_n_search, cell_size);
        rebuild_n_search_tuner = std::make_unique<RebuildNSearchTuner>(total_particles, dev_n_search);
    }

    ~NSearchWrapper() {
        delete_n_search(dev_n_search, host_n_search);
    }

    void rebuild(float *particle_positions, bool tune) {
        clear_n_search(host_n_search);
        rebuild_n_search_tuner->run(particle_positions, tune);
    }

    const NSearch* dev_ptr() const { return dev_n_search; }
    NSearch* dev_ptr() { return dev_n_search; }

private:
    NSearch *dev_n_search;
    NSearch host_n_search;
    std::unique_ptr<RebuildNSearchTuner> rebuild_n_search_tuner;
    unsigned total_particles;
};
