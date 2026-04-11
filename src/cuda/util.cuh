#pragma once

#ifdef NOT_IN_KTT
#include <cuda/math.cuh>
#include <render/box.h>
#endif


struct BoundingBoxGPU {
#ifdef NOT_IN_KTT
    BoundingBoxGPU(const BoundingBox &bb) :
        min{bb.min.x, bb.min.y, bb.min.z, 1.},
        max{bb.max.x, bb.max.y, bb.max.z, 1.},
        model{bb.model}, model_inv{bb.model_inv} {}
#endif

    float4 min, max;
    mat4 model, model_inv;
};

__device__ __host__ inline float4 get_pos(const float *positions, unsigned i) {
    unsigned ii = 3 * i;
    return make_float4(positions[ii], positions[ii + 1], positions[ii + 2], 1.);
}

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
