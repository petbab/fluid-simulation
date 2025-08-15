#include "fluid_simulation.h"

#include <random>
#include <glm/gtc/constants.hpp>


template<int ELEM_SIZE>
static inline auto vec_to_span(const std::vector<glm::vec<ELEM_SIZE, float>> &v) -> std::span<const float> {
    return {reinterpret_cast<const float *>(v.data()), v.size() * ELEM_SIZE};
}

FluidSimulation::FluidSimulation(unsigned int grid_count, BoundingBox bounding_box)
    : kernel{SUPPORT_RADIUS}, bounding_box{bounding_box} {
    init_positions(grid_count);
    init_simulation();
}

auto FluidSimulation::get_position_data() -> std::span<const float> {
    return vec_to_span(positions);
}

void FluidSimulation::init_positions(const unsigned grid_count) {
    assert(grid_count > 1);

    const unsigned particle_count = grid_count * grid_count;// * grid_count;
    const glm::vec3 center = (bounding_box.min + bounding_box.max) / 2.f;
    const glm::vec3 grid_start = center - glm::vec3{static_cast<float>(grid_count - 1)} * PARTICLE_RADIUS + glm::vec3{0.1, 0, 0};

    assert(grid_start.x >= bounding_box.min.x);
    assert(grid_start.y >= bounding_box.min.y);
    assert(grid_start.z >= bounding_box.min.z);

    positions.reserve(particle_count);
    for (unsigned x = 0; x < grid_count; ++x)
        for (unsigned y = 0; y < grid_count; ++y)
//            for (unsigned z = 0; z < grid_count; ++z)
                positions.push_back(grid_start + glm::vec3{
                    static_cast<float>(x) * PARTICLE_SPACING,
                    static_cast<float>(y) * PARTICLE_SPACING, 0.f
//                    static_cast<float>(z) * PARTICLE_SPACING
                });
}

void FluidSimulation::update(double delta) {
    simulation_step(delta);
}
