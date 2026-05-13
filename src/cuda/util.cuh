#pragma once

#ifdef NOT_IN_KTT
#include <cuda/math.cuh>
#include <render/box.h>
#endif


/**
 * @brief Axis-aligned bounding box representation for GPU kernels.
 *
 * Convertible from CPU-side BoundingBox when NOT_IN_KTT is defined.
 */
struct BoundingBoxGPU {
#ifdef NOT_IN_KTT
    /**
     * @brief Converts from CPU BoundingBox.
     * @param bb Source bounding box.
     */
    BoundingBoxGPU(const BoundingBox &bb) :
        min{bb.min.x, bb.min.y, bb.min.z, 1.},
        max{bb.max.x, bb.max.y, bb.max.z, 1.},
        model{bb.model}, model_inv{bb.model_inv} {}
#endif

    float4 min, max;   ///< Minimum and maximum corners.
    mat4 model, model_inv;  ///< Model and inverse model matrices.
};

/**
 * @brief Checks if particle index i belongs to the boundary.
 * @param i Particle index.
 * @param fluid_n Number of fluid particles.
 * @return True if i >= fluid_n.
 */
__device__ __host__ inline bool is_boundary(unsigned i, unsigned fluid_n) {
    return i >= fluid_n;
}

/**
 * @brief Retrieves boundary mass for a particle.
 * @param boundary_mass Device array of boundary masses.
 * @param total_i Absolute particle index.
 * @param fluid_n Number of fluid particles.
 * @return Boundary mass for the particle.
 */
__device__ inline float get_mass(const float *boundary_mass, unsigned total_i, unsigned fluid_n) {
    return boundary_mass[total_i - fluid_n];
}
