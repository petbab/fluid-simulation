#pragma once

#include "nsearch.cuh"
#include "cuda/tuning/rebuild_n_search.cuh"


class NSearchWrapper {
public:
    explicit NSearchWrapper(float cell_size, unsigned total_particles, bool boundary = false)
        : rebuild_n_search_tuner(total_particles, boundary), total_particles(total_particles) {
        dev_n_search = new_n_search(host_n_search, cell_size);
    }

    ~NSearchWrapper() {
        delete_n_search(dev_n_search, host_n_search);
    }

    void rebuild(float4 *particle_positions, bool tune) {
        clear_n_search(host_n_search);
        rebuild_n_search_tuner.run(dev_n_search, particle_positions, total_particles, tune);
    }

    const NSearch* dev_ptr() const { return dev_n_search; }
    NSearch* dev_ptr() { return dev_n_search; }

    void print_stats() const {
        auto host_search = NSearchHost::copy_from_device(dev_n_search);
        int max = 0;
        int min = 10000;
        int sum = 0;
        int count = 0;
        for (int i = 0; i < NSearch::TABLE_SIZE; ++i) {
            if (host_search.table[i] == NSearch::EMPTY_HASH)
                continue;

            int n = host_search.cell_end[i] - host_search.cell_start[i];
            if (n > max) max = n;
            if (n < min) min = n;
            sum += n;
            count++;
        }
        std::cout << "Max: " << max << ", Min: " << min << ", Mean: "
            << static_cast<float>(sum) / static_cast<float>(count) << std::endl;
    }

private:
    NSearch *dev_n_search;
    NSearch host_n_search;
    RebuildNSearchTuner rebuild_n_search_tuner;
    unsigned total_particles;
};
