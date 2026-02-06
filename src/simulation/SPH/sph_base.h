#pragma once

#include <CompactNSearch>
#include <memory>
#include "../fluid_simulator.h"
#include "kernel.h"


class SPHBase : public FluidSimulator {
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

    SPHBase(grid_dims_t grid_dims, const BoundingBox &bounding_box,
        const std::vector<const Object*> &collision_objects, float support_radius);

protected:
    void compute_densities();

    void update_positions(float delta);
    void resolve_collisions();

    void apply_non_pressure_forces(float delta);

    void reset() override;

    void find_neighbors();
    void z_sort();
    void for_neighbors(unsigned i, auto f) {
        CompactNSearch::PointSet &ps = n_search->point_set(point_set_index);
        for (unsigned j = 0; j < ps.n_neighbors(point_set_index, i); ++j)
            f(ps.neighbor(point_set_index, i, j));
    }

    // Adapt the time step size according to the Courant-Friedrich-Levy (CFL) condition
    float adapt_time_step(float delta, float min_step, float max_step, float h) const {
        float max_velocity = glm::length(std::ranges::max(velocities, std::less{}, [](const glm::vec3 &v){
            return glm::length(v);
        }));

        if (max_velocity < 1.e-9)
            return max_step;

        float cfl_max_time_step = CFL_FACTOR * h / max_step;

        return std::min(std::clamp(delta, min_step, max_step), cfl_max_time_step);
    }

private:
    /**
     * Simulates viscosity by smoothing the velocity field [SPH Tutorial, eq. 103].
     */
    void compute_XSPH();

    /**
     * Computes and applies the viscous force using an explicit viscosity model.
     * Approximates the Laplacian of the velocity field via finite differences
     * [SPH Tutorial, eq. 102].
     */
    void compute_viscosity();

    /**
     * Simulates surface tension using the macroscopic approach by Akinci et al. (2013).
     */
    void compute_surface_tension();

    /**
     * Computes surface normals used to calculate the curvature force
     * in compute_surface_tension [SPH Tutorial, eq. 125].
     */
    void compute_surface_normals();

protected:
    std::vector<glm::vec3> velocities, non_pressure_accel, normals;
    std::vector<float> densities;

private:
    std::unique_ptr<CompactNSearch::NeighborhoodSearch> n_search;
    unsigned point_set_index;
    CubicSpline cubic_k;
    CohesionKernel cohesion_k;
};
