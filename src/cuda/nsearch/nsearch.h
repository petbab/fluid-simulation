#pragma once

#include "nsearch.cuh"
#include "cuda/tuning/rebuild_n_search.cuh"


/**
 * @brief Host-side manager for a GPU neighbor search structure.
 *
 * Wraps NSearch allocation, deallocation, and shallow-copy operations.
 */
class NSearchWrapper {
public:
    /**
     * @brief Constructs and allocates the device neighbor search.
     * @param table_size Number of hash table slots.
     * @param cell_size Spatial cell size.
     * @param particle_n Total number of particles.
     */
    explicit NSearchWrapper(unsigned table_size, float cell_size, unsigned particle_n)
        : particle_n(particle_n) {
        dev_n_search = new_n_search(host_n_search, table_size, cell_size);
    }

    virtual ~NSearchWrapper() {
        delete_n_search(dev_n_search, host_n_search);
    }

    /** @brief Clears the hash table. */
    void clear() { clear_n_search(host_n_search); }

    /**
     * @brief Shallow-copies host data to a device NSearch.
     * @param dev_dst Destination device pointer.
     */
    void shallow_copy(NSearch *dev_dst) const {
        shallow_copy_n_search(dev_dst, host_n_search);
    }

    /** @return Const device pointer. */
    const NSearch* dev_ptr() const { return dev_n_search; }
    /** @return Device pointer. */
    NSearch* dev_ptr() { return dev_n_search; }

    /** @brief Prints cell occupancy statistics. */
    void print_stats() const {
        auto host_search = NSearchHost::copy_from_device(dev_n_search);
        int max = 0;
        int min = 10000;
        int sum = 0;
        int count = 0;
        for (int i = 0; i < host_search.table_size; ++i) {
            if (host_search.table[i] == NSearch::EMPTY_HASH)
                continue;

            int n = host_search.cell_end[i] - host_search.cell_start[i];
            if (n > max) max = n;
            if (n < min) min = n;
            sum += n;
            count++;
        }
        std::cout << "Max: " << max << ", Min: " << min << ", Count: " << count << ", Mean: "
            << static_cast<float>(sum) / static_cast<float>(count) << std::endl;
    }

protected:
    NSearch *dev_n_search;   ///< Device NSearch pointer.
    NSearch host_n_search;   ///< Host NSearch descriptor.
    unsigned particle_n;     ///< Total particle count.
};

/**
 * @brief Tuned variant of NSearchWrapper that uses KTT-tuned rebuild kernels.
 */
class NSearchWrapperTuned : public NSearchWrapper {
public:
    /**
     * @brief Constructs the tuned neighbor search wrapper.
     * @param table_size Number of hash table slots.
     * @param cell_size Spatial cell size.
     * @param particle_n Total number of particles.
     * @param boundary If true, uses boundary-specific tuning.
     */
    explicit NSearchWrapperTuned(unsigned table_size, float cell_size, unsigned particle_n, bool boundary = false)
        : NSearchWrapper(table_size, cell_size, particle_n), rebuild_n_search_tuner(particle_n, boundary) {}

    /**
     * @brief Rebuilds the neighbor search hash table.
     * @param particle_positions Device pointer to particle positions.
     * @param tune If true, runs the KTT tuner.
     */
    void rebuild(float4 *particle_positions, bool tune) {
        clear();
        rebuild_n_search_tuner.run(dev_n_search, particle_positions, particle_n, tune);
    }

private:
    RebuildNSearchTuner rebuild_n_search_tuner;  ///< Tuner for rebuild kernel.
};
