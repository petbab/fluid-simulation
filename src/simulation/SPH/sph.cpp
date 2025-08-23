#include "sph.h"
#include <algorithm>
#include <numeric>


SPHSimulator::SPHSimulator(unsigned int grid_count, BoundingBox bounding_box, bool is_2d)
    : SPHBase(grid_count, bounding_box, SUPPORT_RADIUS, is_2d),
      cubic_k{SUPPORT_RADIUS, is_2d}, spiky_k{SUPPORT_RADIUS} {
    velocities.resize(positions.size());
    pressure.resize(positions.size());
    pressure_accel.resize(positions.size());
}

void SPHSimulator::update(double delta) {
    delta = std::clamp(delta, static_cast<double>(MIN_TIME_STEP), static_cast<double>(MAX_TIME_STEP));

    compute_densities(PARTICLE_MASS, cubic_k);

    apply_non_pressure_forces(delta);

    compute_pressure();
    apply_pressure_force(delta);

    update_positions(delta);
    resolve_collisions();
    find_neighbors();
}

void SPHSimulator::apply_non_pressure_forces(double delta) {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < velocities.size(); ++i)
        velocities[i] += static_cast<float>(delta) * (GRAVITY + compute_viscosity(i));
}

glm::vec3 SPHSimulator::compute_viscosity(unsigned i) {
    glm::vec3 velocity_laplacian{0.f};
    glm::vec3 xi = positions[i];
    glm::vec3 vi = velocities[i];

    for_neighbors(i, [&](unsigned j){
        glm::vec3 x_ij = xi - positions[j];
        glm::vec3 v_ij = vi - velocities[j];

        velocity_laplacian += glm::dot(v_ij, x_ij) * cubic_k.grad_W(x_ij) / (densities[j] * (glm::dot(x_ij, x_ij) + 0.01f * SUPPORT_RADIUS * SUPPORT_RADIUS));
    });

    velocity_laplacian *= 10 * PARTICLE_MASS;

    return VISCOSITY * velocity_laplacian;
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

void SPHSimulator::update_positions(double delta) {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < positions.size(); ++i)
        positions[i] += static_cast<float>(delta) * velocities[i];
}

void SPHSimulator::resolve_collisions() {
    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < positions.size(); ++i) {
        if (positions[i].x - PARTICLE_RADIUS < bounding_box.min.x) {
            positions[i].x = bounding_box.min.x + PARTICLE_RADIUS;
            velocities[i].x *= -ELASTICITY;
        } else if (positions[i].x + PARTICLE_RADIUS > bounding_box.max.x) {
            positions[i].x = bounding_box.max.x - PARTICLE_RADIUS;
            velocities[i].x *= -ELASTICITY;
        }
        if (positions[i].y - PARTICLE_RADIUS < bounding_box.min.y) {
            positions[i].y = bounding_box.min.y + PARTICLE_RADIUS;
            velocities[i].y *= -ELASTICITY;
        } else if (positions[i].y + PARTICLE_RADIUS > bounding_box.max.y) {
            positions[i].y = bounding_box.max.y - PARTICLE_RADIUS;
            velocities[i].y *= -ELASTICITY;
        }
        if (positions[i].z - PARTICLE_RADIUS < bounding_box.min.z) {
            positions[i].z = bounding_box.min.z + PARTICLE_RADIUS;
            velocities[i].z *= -ELASTICITY;
        } else if (positions[i].z + PARTICLE_RADIUS > bounding_box.max.z) {
            positions[i].z = bounding_box.max.z - PARTICLE_RADIUS;
            velocities[i].z *= -ELASTICITY;
        }
    }
}
