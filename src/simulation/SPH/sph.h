#pragma once

#include "common.h"


class SPHSimulator final : public SPHBase {
public:
    ///////////////////////////////////////////////////////////////////////////////
    ////                         SIMULATION PARAMETERS                         ////
    ///////////////////////////////////////////////////////////////////////////////
    static constexpr float REST_DENSITY = 1000.f;
    static constexpr float SUPPORT_RADIUS = 2.f * PARTICLE_SPACING;
    static constexpr float PARTICLE_VOLUME = PARTICLE_SPACING * PARTICLE_SPACING * PARTICLE_SPACING * 0.8;
    static constexpr float PARTICLE_MASS = REST_DENSITY * PARTICLE_VOLUME;

    static constexpr glm::vec3 GRAVITY{0, -9.81f, 0};
    static constexpr float VISCOSITY = 0.01f;

    static constexpr float STIFFNESS = 1.f;
    static constexpr float EXPONENT = 3.f;

    static constexpr float MAX_TIME_STEP = 0.0005f;
    static constexpr float MIN_TIME_STEP = 0.00001f;
    ///////////////////////////////////////////////////////////////////////////////

    SPHSimulator(unsigned grid_count, BoundingBox bounding_box, bool is_2d = false);

    void update(double delta) override;

private:
    void apply_non_pressure_forces(double delta);
    glm::vec3 compute_viscosity(unsigned i);

    void compute_pressure();
    void apply_pressure_force(double delta);

    void update_positions(double delta);

    std::vector<float> pressure;

    CubicSpline cubic_k;
    SpikyKernel spiky_k;
};
