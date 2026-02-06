#include "sph.h"
#include <algorithm>


SPHSimulator::SPHSimulator(grid_dims_t grid_dims, const BoundingBox &bounding_box,
    const std::vector<const Object*> &collision_objects)
    : SPHBase(grid_dims, bounding_box, collision_objects, SUPPORT_RADIUS), spiky_k{SUPPORT_RADIUS} {
    pressure.resize(fluid_particles);
}

void SPHSimulator::update(float delta) {
    compute_densities();

    apply_non_pressure_forces(delta);

    delta = adapt_time_step(delta, MIN_TIME_STEP, MAX_TIME_STEP, SUPPORT_RADIUS);

    compute_pressure();
    apply_pressure_force(delta);

    update_positions(delta);
    resolve_collisions();
    find_neighbors();
}

void SPHSimulator::compute_pressure() {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < fluid_particles; ++i) {
        float d = std::max(densities[i], REST_DENSITY);
        pressure[i] = STIFFNESS * (std::pow(d / REST_DENSITY, EXPONENT) - 1.f);
    }
}

void SPHSimulator::apply_pressure_force(float delta) {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < fluid_particles; ++i) {
        glm::vec3 p_accel{0.f};
        glm::vec3 xi = positions[i];
        float dpi = pressure[i] / (densities[i] * densities[i]);

        for_neighbors(i, [&](unsigned j){
            float dpj = pressure[j] / (densities[j] * densities[j]);
            p_accel -= (dpi + dpj) * spiky_k.grad_W(xi - positions[j]);
        });

        velocities[i] += delta * PARTICLE_MASS * p_accel;
    }
}

void SPHSimulator::reset() {
    SPHBase::reset();
    std::ranges::fill(pressure, 0);
}
