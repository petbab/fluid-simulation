#include "common.h"


SPHBase::SPHBase(unsigned int grid_count, const BoundingBox &bounding_box, float support_radius, bool is_2d)
    : FluidSimulator{grid_count, bounding_box, is_2d},
      n_search{std::make_unique<CompactNSearch::NeighborhoodSearch>(support_radius)},
      cubic_k{SUPPORT_RADIUS, is_2d},
      cohesion_k{SUPPORT_RADIUS} {
    densities.resize(positions.size());
    velocities.resize(positions.size());
    non_pressure_accel.resize(positions.size());
    normals.resize(positions.size());

    point_set_index = n_search->add_point_set(reinterpret_cast<float*>(positions.data()), positions.size());
    z_sort();
    find_neighbors();
}

void SPHBase::z_sort() {
    n_search->z_sort();
    CompactNSearch::PointSet &ps = n_search->point_set(point_set_index);
    ps.sort_field(positions.data());
}

void SPHBase::compute_densities() {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < densities.size(); ++i) {
        float density = cubic_k.W(glm::vec3{0.f});
        glm::vec3 xi = positions[i];

        for_neighbors(i, [&](unsigned j){
            density += cubic_k.W(xi - positions[j]);
        });

        densities[i] = density * PARTICLE_MASS;
    }
}

void SPHBase::update_positions(double delta) {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < positions.size(); ++i)
        positions[i] += static_cast<float>(delta) * velocities[i];
}

void SPHBase::resolve_collisions() {
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

void SPHBase::apply_non_pressure_forces(double delta) {
    std::ranges::fill(non_pressure_accel, GRAVITY);

    compute_viscosity();
    compute_surface_tension();

    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < velocities.size(); ++i)
        velocities[i] += static_cast<float>(delta) * non_pressure_accel[i];

    compute_XSPH();
}

void SPHBase::compute_XSPH() {
    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < velocities.size(); ++i) {
        glm::vec3 vi = velocities[i];
        glm::vec3 xi = positions[i];
        glm::vec3 sum{0.f};

        for_neighbors(i, [&](unsigned j){
            sum += (velocities[j] - vi) * cubic_k.W(xi - positions[j]) / densities[j];
        });

        velocities[i] += XSPH_ALPHA * PARTICLE_MASS * sum;
    }
}

void SPHBase::compute_viscosity() {
    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < positions.size(); ++i) {
        glm::vec3 velocity_laplacian{0.f};
        glm::vec3 xi = positions[i];
        glm::vec3 vi = velocities[i];

        for_neighbors(i, [&](unsigned j){
            glm::vec3 x_ij = xi - positions[j];
            glm::vec3 v_ij = vi - velocities[j];

            velocity_laplacian += glm::dot(v_ij, x_ij) * cubic_k.grad_W(x_ij) / (densities[j] * (glm::dot(x_ij, x_ij) + 0.01f * SUPPORT_RADIUS * SUPPORT_RADIUS));
        });

        velocity_laplacian *= 10 * PARTICLE_MASS;

        non_pressure_accel[i] += VISCOSITY * velocity_laplacian;
    }
}

void SPHBase::compute_surface_tension() {
    compute_surface_normals();

    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < positions.size(); ++i) {
        glm::vec3 xi = positions[i];
        glm::vec3 ni = normals[i];
        glm::vec3 f{0.f};

        for_neighbors(i, [&](unsigned j) {
            glm::vec3 x_ij = xi - positions[j];
            if (glm::dot(x_ij, x_ij) > 1e-6)
                f += (PARTICLE_MASS * glm::normalize(x_ij) * cohesion_k.W(x_ij) + ni - normals[j])
                    / (densities[i] + densities[j]);
        });

        non_pressure_accel[i] -= SURFACE_TENSION_ALPHA * 2.f * REST_DENSITY * f;
    }
}

void SPHBase::compute_surface_normals() {
    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < normals.size(); ++i) {
        glm::vec3 xi = positions[i];
        glm::vec3 n{0.f};

        for_neighbors(i, [&](unsigned j){
            n += cubic_k.grad_W(xi - positions[j]) / densities[j];
        });

        normals[i] = SUPPORT_RADIUS * PARTICLE_MASS * n;
    }
}

void SPHBase::reset() {
    FluidSimulator::reset();

    std::ranges::fill(velocities, glm::vec3{0});
    std::ranges::fill(non_pressure_accel, glm::vec3{0});
    std::ranges::fill(densities, 0);

    z_sort();
    find_neighbors();
}
