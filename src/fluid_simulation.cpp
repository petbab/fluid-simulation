#include "fluid_simulation.h"

#include <random>
#include <glm/gtc/constants.hpp>


static constexpr float RADIUS = 0.1;
static constexpr float ELASTICITY = 0.8;

template<int ELEM_SIZE>
static inline auto vec_to_span(const std::vector<glm::vec<ELEM_SIZE, float>> &v) -> std::span<const float> {
    return {reinterpret_cast<const float *>(v.data()), v.size() * ELEM_SIZE};
}

FluidSimulation::FluidSimulation(unsigned int grid_count, float gap, BoundingBox bounding_box)
    : bounding_box{bounding_box} {
    initiate_particles(grid_count, gap);
}

void FluidSimulation::update(double delta) {
    for (unsigned i = 0; i < positions.size(); ++i) {
        velocities[i] += glm::vec3{0, -9.81 * delta, 0};

        glm::vec3 new_pos = positions[i] + velocities[i] * static_cast<float>(delta);
        if (new_pos.x - RADIUS < bounding_box.min.x) {
            new_pos.x = bounding_box.min.x + RADIUS;
            velocities[i].x *= -ELASTICITY;
        } else if (new_pos.x + RADIUS > bounding_box.max.x) {
            new_pos.x = bounding_box.max.x - RADIUS;
            velocities[i].x *= -ELASTICITY;
        }
        if (new_pos.y - RADIUS < bounding_box.min.y) {
            new_pos.y = bounding_box.min.y + RADIUS;
            velocities[i].y *= -ELASTICITY;
        } else if (new_pos.y + RADIUS > bounding_box.max.y) {
            new_pos.y = bounding_box.max.y - RADIUS;
            velocities[i].y *= -ELASTICITY;
        }
        if (new_pos.z - RADIUS < bounding_box.min.z) {
            new_pos.z = bounding_box.min.z + RADIUS;
            velocities[i].z *= -ELASTICITY;
        } else if (new_pos.z + RADIUS > bounding_box.max.z) {
            new_pos.z = bounding_box.max.z - RADIUS;
            velocities[i].z *= -ELASTICITY;
        }
        positions[i] = new_pos;
    }
}

auto FluidSimulation::get_positions() -> std::span<const float> {
    return vec_to_span(positions);
}

static glm::vec3 random_direction() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Generate random spherical coordinates
    float theta = dis(gen) * 2.0f * glm::pi<float>(); // Azimuth: [0, 2π]
    float phi = std::acos(2.0f * dis(gen) - 1.0f);    // Polar angle: [0, π] with uniform distribution on sphere

    // Convert to Cartesian coordinates
    float sinPhi = std::sin(phi);
    return {
        sinPhi * std::cos(theta),
        sinPhi * std::sin(theta),
        std::cos(phi)
    };
}

void FluidSimulation::initiate_particles(const unsigned grid_count, const float gap) {
    assert(grid_count > 1);

    const unsigned particle_count = grid_count * grid_count * grid_count;
    const glm::vec3 center = (bounding_box.min + bounding_box.max) / 2.f;
    const glm::vec3 grid_start = center - static_cast<float>(grid_count - 1) * gap / 2.f;

    assert(grid_start.x >= bounding_box.min.x);
    assert(grid_start.y >= bounding_box.min.y);
    assert(grid_start.z >= bounding_box.min.z);

    positions.reserve(particle_count);
    for (unsigned x = 0; x < grid_count; ++x)
        for (unsigned y = 0; y < grid_count; ++y)
            for (unsigned z = 0; z < grid_count; ++z)
                positions.push_back(grid_start + glm::vec3{
                    static_cast<float>(x) * gap,
                    static_cast<float>(y) * gap,
                    static_cast<float>(z) * gap
                });

    velocities.reserve(particle_count);
    for (unsigned i = 0; i < particle_count; ++i)
        velocities.push_back(random_direction());
}
