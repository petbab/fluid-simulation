#include "sph.h"
#include <algorithm>


SPHSimulator::SPHSimulator(unsigned int grid_count, const BoundingBox &bounding_box, bool is_2d)
    : SPHBase(grid_count, bounding_box, SUPPORT_RADIUS, is_2d), spiky_k{SUPPORT_RADIUS} {
    pressure.resize(positions.size());
}

void SPHSimulator::update(double delta) {
    delta = adapt_time_step<MIN_TIME_STEP, MAX_TIME_STEP, SUPPORT_RADIUS>(delta);

    compute_densities();

    apply_non_pressure_forces(delta);

    compute_pressure();
    apply_pressure_force(delta);

    update_positions(delta);
    resolve_collisions();
    find_neighbors();
}

void SPHSimulator::compute_pressure() {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < pressure.size(); ++i) {
        float d = std::max(densities[i], REST_DENSITY);
        pressure[i] = STIFFNESS * (std::pow(d / REST_DENSITY, EXPONENT) - 1.f);
    }
}

void SPHSimulator::apply_pressure_force(double delta) {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < velocities.size(); ++i) {
        glm::vec3 p_accel{0.f};
        glm::vec3 xi = positions[i];
        float dpi = pressure[i] / (densities[i] * densities[i]);

        for_neighbors(i, [&](unsigned j){
            float dpj = pressure[j] / (densities[j] * densities[j]);
            p_accel -= (dpi + dpj) * spiky_k.grad_W(xi - positions[j]);
        });

        velocities[i] += static_cast<float>(delta) * PARTICLE_MASS * p_accel;
    }
}

void SPHSimulator::reset() {
    SPHBase::reset();
    std::ranges::fill(pressure, 0);
}
