#pragma once

#include <thrust/device_vector.h>


struct NeighborList {
    NeighborList(unsigned particle_count, unsigned max_neighbors)
        : max_neighbors(max_neighbors), particle_count(particle_count),
        capacity(static_cast<unsigned long long>(particle_count) * max_neighbors),
        counts_vec(particle_count),
        offsets_vec(particle_count),
        neighbors_vec(capacity) {}

    unsigned *counts() { return dev_ptr(counts_vec); }
    unsigned *offsets() { return dev_ptr(offsets_vec); }
    unsigned *neighbors() { return dev_ptr(neighbors_vec); }

    const unsigned *counts() const { return dev_ptr(counts_vec); }
    const unsigned *offsets() const { return dev_ptr(offsets_vec); }
    const unsigned *neighbors() const { return dev_ptr(neighbors_vec); }

    const unsigned max_neighbors, particle_count;
    const unsigned long long capacity;
    thrust::device_vector<unsigned> counts_vec, offsets_vec, neighbors_vec;

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
