#pragma once

#include <cuda_runtime.h>


template<class F>
__device__ void device_check(F f) {
    f(float4{0.});
}

template<class F>
concept DeviceCallable = requires(F f) {
    device_check(f);
};

__device__ inline void set_pos(float *positions, unsigned i, float4 pos) {
    unsigned ii = 3 * i;
    positions[ii] = pos.x;
    positions[ii + 1] = pos.y;
    positions[ii + 2] = pos.z;
}

__device__ __host__ inline bool is_boundary(unsigned i, unsigned fluid_n) {
    return i >= fluid_n;
}

__device__ inline float get_mass(const float *boundary_mass, unsigned total_i, unsigned fluid_n) {
    return boundary_mass[total_i - fluid_n];
}
