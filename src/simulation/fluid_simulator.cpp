#include "fluid_simulator.h"

#include <random>
#include <glm/gtc/constants.hpp>


template<int ELEM_SIZE>
static inline auto vec_to_span(const std::vector<glm::vec<ELEM_SIZE, float>> &v) -> std::span<const float> {
    return {reinterpret_cast<const float *>(v.data()), v.size() * ELEM_SIZE};
}

FluidSimulator::FluidSimulator(unsigned int grid_count, const BoundingBox &bounding_box, bool is_2d)
    : bounding_box{bounding_box}, grid_count{grid_count}, is_2d{is_2d} {
    init_positions();
}

auto FluidSimulator::get_position_data() -> std::span<const float> {
    return vec_to_span(positions);
}

void FluidSimulator::init_positions() {
    assert(grid_count > 1);

    const unsigned particle_count = grid_count * grid_count * (is_2d ? 1 : grid_count);
    const glm::vec3 center = (bounding_box.min + bounding_box.max) / 2.f;
    const glm::vec3 grid_start = center - glm::vec3{static_cast<float>(grid_count - 1)} * PARTICLE_RADIUS + glm::vec3{0.1, -0.1, 0};

    assert(grid_start.x >= bounding_box.min.x);
    assert(grid_start.y >= bounding_box.min.y);
    assert(grid_start.z >= bounding_box.min.z);

    positions.resize(particle_count);
    unsigned i = 0;
    for (unsigned x = 0; x < grid_count; ++x)
        for (unsigned y = 0; y < grid_count; ++y) {
            if (!is_2d) {
                for (unsigned z = 0; z < grid_count; ++z) {
                    positions[i] = grid_start + glm::vec3{
                        static_cast<float>(x) * PARTICLE_SPACING,
                        static_cast<float>(y) * PARTICLE_SPACING,
                        static_cast<float>(z) * PARTICLE_SPACING
                    };
                    ++i;
                }
            } else {
                positions[i] = grid_start + glm::vec3{
                    static_cast<float>(x) * PARTICLE_SPACING,
                    static_cast<float>(y) * PARTICLE_SPACING,
                    center.z
                };
                ++i;
            }
        }
}

void FluidSimulator::reset() {
    init_positions();
}
