#include "common.h"


SPHBase::SPHBase(unsigned int grid_count, BoundingBox bounding_box, float support_radius, bool is_2d)
    : FluidSimulator{grid_count, bounding_box, is_2d},
      n_search{std::make_unique<CompactNSearch::NeighborhoodSearch>(support_radius)} {
    densities.resize(positions.size());
    velocities.resize(positions.size());
    XSPH_accel.resize(positions.size());

    point_set_index = n_search->add_point_set(reinterpret_cast<float*>(positions.data()), positions.size());
    z_sort();
    find_neighbors();
}

void SPHBase::z_sort() {
    n_search->z_sort();
    CompactNSearch::PointSet &ps = n_search->point_set(point_set_index);
    ps.sort_field(positions.data());
}

void SPHBase::compute_densities(float particle_mass, const Kernel &kernel) {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < densities.size(); ++i) {
        float density = kernel.W(glm::vec3{0.f});
        glm::vec3 xi = positions[i];

        for_neighbors(i, [&](unsigned j){
            density += kernel.W(xi - positions[j]);
        });

        densities[i] = density * particle_mass;
    }
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

void SPHBase::apply_XSPH(const Kernel &kernel, float particle_mass) {
    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < velocities.size(); ++i) {
        glm::vec3 vi = velocities[i];
        glm::vec3 xi = positions[i];
        glm::vec3 sum{0.f};

        for_neighbors(i, [&](unsigned j){
            sum += (velocities[j] - vi) * kernel.W(xi - positions[j]) / densities[j];
        });

        XSPH_accel[i] = XSPH_ALPHA * particle_mass * sum;
    }

    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < velocities.size(); ++i)
        velocities[i] += XSPH_accel[i];
}

void SPHBase::reset() {
    FluidSimulator::reset();

    std::ranges::fill(velocities, glm::vec3{0});
    std::ranges::fill(XSPH_accel, glm::vec3{0});
    std::ranges::fill(densities, 0);

    z_sort();
    find_neighbors();
}
