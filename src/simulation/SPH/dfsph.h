#pragma once

#include "common.h"
#include <vector>
#include <glm/vec3.hpp>
#include <glm/gtc/constants.hpp>
#include <span>


class DFSPHSimulator final : public SPHBase {
public:
    ///////////////////////////////////////////////////////////////////////////////
    ////                         SIMULATION PARAMETERS                         ////
    ///////////////////////////////////////////////////////////////////////////////
    static constexpr float REST_DENSITY = 1000.f;
    static constexpr float SUPPORT_RADIUS = 2.f * PARTICLE_SPACING;
    static constexpr float PARTICLE_VOLUME = PARTICLE_SPACING * PARTICLE_SPACING * PARTICLE_SPACING;
    static constexpr float PARTICLE_MASS = REST_DENSITY * PARTICLE_VOLUME;

    static constexpr float ALPHA_DENOM_EPSILON = 1.e-5f;

    static constexpr glm::vec3 GRAVITY{0, -9.81f, 0};
    static constexpr float ELASTICITY = 0.9f;
    static constexpr float VISCOSITY = 0.01f;

    static constexpr float MAX_TIME_STEP = 0.001f;
    static constexpr float MIN_TIME_STEP = 0.0001f;
    static constexpr float CFL_FACTOR = 0.4f;

    static constexpr float MAX_DIVERGENCE_ERROR = 0.1f; // 0.001 * REST_DENSITY
    static constexpr float MAX_DENSITY_ERROR = 0.001f; // 0.001 * REST_DENSITY
    static constexpr int MAX_DIVERGENCE_ITERATIONS = 100;
    static constexpr int MAX_DENSITY_ITERATIONS = 100;
    ///////////////////////////////////////////////////////////////////////////////

    DFSPHSimulator(unsigned grid_count, BoundingBox bounding_box);

    void update(double delta) override;

private:
    void compute_alphas();

    // Adapt the time step size according to the Courant-Friedrich-Levy (CFL) condition
    double adapt_time_step(double delta) const;
    void predict_velocities(double delta);
    void update_positions(double delta);
    void correct_density_error(double delta);
    void correct_divergence_error(double delta);
    void warm_start_density(double delta);
    void warm_start_divergence(double delta);

    void resolve_collisions(double delta);

    std::vector<glm::vec3> velocities;
    std::vector<float> predicted_densities,
        alphas, divergence_errors,
        divergence_kappas, density_kappas;

    CubicSpline kernel;

    bool first_iteration = true;
};
