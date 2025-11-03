#pragma once

#include <glm/glm.hpp>

#include <thrust/device_vector.h>
#include "../simulator.h"


__device__ inline glm::vec3 get_pos(const float *positions, unsigned i) {
    unsigned ii = 3 * i;
    return {positions[ii], positions[ii + 1], positions[ii + 2]};
}

__device__ inline void set_pos(float *positions, unsigned i, glm::vec3 pos) {
    unsigned ii = 3 * i;
    positions[ii] = pos.x;
    positions[ii + 1] = pos.y;
    positions[ii + 2] = pos.z;
}

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

    CUDASPHBase(unsigned grid_count, const BoundingBox &bounding_box, bool is_2d = false);

protected:
    void compute_densities(const float *positions_dev_ptr);
    void update_positions(float *positions_dev_ptr, float delta);
    void resolve_collisions(float *positions_dev_ptr);
    void apply_non_pressure_forces(float delta);

    void reset() override;

    thrust::device_vector<float> density;
    thrust::device_vector<glm::vec3> velocity;

private:
    thrust::device_vector<glm::vec3> non_pressure_accel;
};
