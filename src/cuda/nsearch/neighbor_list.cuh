#pragma once

#include <thrust/device_vector.h>


/**
 * @brief GPU neighbor list storage for SPH particles.
 *
 * Maintains per-particle neighbor counts, offsets into a flat neighbor array,
 * and the neighbor indices themselves. All data lives in device memory.
 */
struct NeighborList {
    /**
     * @brief Constructs a neighbor list with given capacity.
     * @param particle_count Number of fluid particles.
     * @param max_neighbors Maximum neighbors per particle.
     */
    NeighborList(unsigned particle_count, unsigned max_neighbors)
        : max_neighbors(max_neighbors), particle_count(particle_count),
        capacity(static_cast<unsigned long long>(particle_count) * max_neighbors),
        counts_vec(particle_count),
        offsets_vec(particle_count),
        neighbors_vec(capacity) {}

    /** @return Device pointer to per-particle neighbor counts. */
    unsigned *counts() { return dev_ptr(counts_vec); }
    /** @return Device pointer to per-particle neighbor offsets. */
    unsigned *offsets() { return dev_ptr(offsets_vec); }
    /** @return Device pointer to flat neighbor indices array. */
    unsigned *neighbors() { return dev_ptr(neighbors_vec); }

    /** @return Const device pointer to per-particle neighbor counts. */
    const unsigned *counts() const { return dev_ptr(counts_vec); }
    /** @return Const device pointer to per-particle neighbor offsets. */
    const unsigned *offsets() const { return dev_ptr(offsets_vec); }
    /** @return Const device pointer to flat neighbor indices array. */
    const unsigned *neighbors() const { return dev_ptr(neighbors_vec); }

    /** @brief Prints neighbor count statistics to stdout. */
    void print_stats() const {
        auto [min_it, max_it] = thrust::minmax_element(counts_vec.begin(), counts_vec.end());
        long max = *max_it;
        long min = *min_it;
        long sum = thrust::reduce(counts_vec.begin(), counts_vec.end());
        long count = particle_count;
        std::cout << "Max: " << max << ", Min: " << min << ", Count: " << count << ", Mean: "
            << static_cast<float>(sum) / static_cast<float>(count) << std::endl;
    }

    const unsigned max_neighbors, particle_count;  ///< Capacity parameters.
    const unsigned long long capacity;             ///< Total neighbor slot capacity.
    thrust::device_vector<unsigned> counts_vec, offsets_vec, neighbors_vec;  ///< Device storage.

private:
    template <class T>
    static T* dev_ptr(thrust::device_vector<T>& vec) {
        return thrust::raw_pointer_cast(vec.data());
    }
    template <class T>
    static const T* dev_ptr(const thrust::device_vector<T>& vec) {
        return thrust::raw_pointer_cast(vec.data());
    }
};
