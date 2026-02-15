#pragma once

#include <glm/glm.hpp>

#include <thrust/device_vector.h>
#include "../simulator.h"
#include "../nsearch/nsearch.h"


class CUDASPHBase : public CUDASimulator {
public:
    ///////////////////////////////////////////////////////////////////////////////
    ////                         SIMULATION PARAMETERS                         ////
    ///////////////////////////////////////////////////////////////////////////////
    static constexpr float REST_DENSITY = 1000.f;
    static constexpr float SUPPORT_RADIUS = 2.f * PARTICLE_SPACING;
    static constexpr float PARTICLE_VOLUME = PARTICLE_SPACING * PARTICLE_SPACING * PARTICLE_SPACING * 0.8;
    static constexpr float PARTICLE_MASS = REST_DENSITY * PARTICLE_VOLUME;

    static constexpr float ELASTICITY = 0.9f;
    static constexpr glm::vec3 GRAVITY{0, -9.81f, 0};

    static constexpr float XSPH_ALPHA = 0.f;
    static constexpr float VISCOSITY = 0.001f;
    static constexpr float SURFACE_TENSION_ALPHA = 0.15f;

    static constexpr float CFL_FACTOR = 0.4f;
    static constexpr float NON_PRESSURE_MAX_TIME_STEP = 0.015;
    ///////////////////////////////////////////////////////////////////////////////

    CUDASPHBase(const opts_t &opts);

protected:
    void compute_densities(const float *positions_dev_ptr);
    void update_positions(float *positions_dev_ptr, float delta);
    void apply_non_pressure_forces(const float* positions_dev_ptr, float delta);

    void reset() override;

private:
    void update_velocities(float delta);

    /**
     * Simulates viscosity by smoothing the velocity field [SPH Tutorial, eq. 103].
     */
    void compute_XSPH(const float* positions_dev_ptr);

    /**
     * Computes and applies the viscous force using an explicit viscosity model.
     * Approximates the Laplacian of the velocity field via finite differences
     * [SPH Tutorial, eq. 102].
     */
    void compute_viscosity(const float* positions_dev_ptr);

    /**
     * Simulates surface tension using the macroscopic approach by Akinci et al. (2013).
     */
    void compute_surface_tension(const float* positions_dev_ptr);

    /**
     * Computes surface normals used to calculate the curvature force
     * in compute_surface_tension [SPH Tutorial, eq. 125].
     */
    void compute_surface_normals(const float* positions_dev_ptr);

protected:
    thrust::device_vector<float> density;
    thrust::device_vector<glm::vec3> velocity;
    NSearchWrapper n_search;

private:
    thrust::device_vector<glm::vec3> non_pressure_accel, normal;
};

// __device__ inline glm::vec3 get_pos(const float *positions, unsigned i) {
//     unsigned ii = 3 * i;
//     return {positions[ii], positions[ii + 1], positions[ii + 2]};
// }

__device__ inline void set_pos(float *positions, unsigned i, glm::vec3 pos) {
    unsigned ii = 3 * i;
    positions[ii] = pos.x;
    positions[ii + 1] = pos.y;
    positions[ii + 2] = pos.z;
}

__device__ inline bool is_neighbor(glm::vec3 xi, glm::vec3 xj, unsigned i, unsigned j) {
    glm::vec3 r = xi - xj;
    return i != j && glm::dot(r, r) <= CUDASPHBase::SUPPORT_RADIUS * CUDASPHBase::SUPPORT_RADIUS;
}
