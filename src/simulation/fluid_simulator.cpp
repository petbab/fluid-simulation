#include "fluid_simulator.h"

#include <random>


template<int ELEM_SIZE>
static inline auto vec_to_span(const std::vector<glm::vec<ELEM_SIZE, float>> &v) -> std::span<const float> {
    return {reinterpret_cast<const float *>(v.data()), v.size() * ELEM_SIZE};
}

FluidSimulator::FluidSimulator(const opts_t &opts)
    : particle_count{opts.grid_dims.x * opts.grid_dims.y * opts.grid_dims.z},
      bounding_box{opts.bounding_box}, grid_dims{opts.grid_dims} {
    init_positions();
}

auto FluidSimulator::get_position_data() -> std::span<const float> {
    return vec_to_span(positions);
}

void FluidSimulator::init_positions() {
    assert(grid_dims.x > 1);
    assert(grid_dims.y > 1);
    assert(grid_dims.z > 1);

    const glm::vec3 center = (bounding_box.min + bounding_box.max) / 2.f;
    const glm::vec3 grid_start = center - glm::vec3{
            static_cast<float>(grid_dims.x - 1),
            static_cast<float>(grid_dims.y - 1),
            static_cast<float>(grid_dims.z - 1)
        } * PARTICLE_RADIUS + glm::vec3{0.1, -0.1, 0};

    assert(grid_start.x >= bounding_box.min.x);
    assert(grid_start.y >= bounding_box.min.y);
    assert(grid_start.z >= bounding_box.min.z);

    positions.resize(particle_count);
    unsigned i = 0;
    for (unsigned x = 0; x < grid_dims.x; ++x)
    for (unsigned y = 0; y < grid_dims.y; ++y)
    for (unsigned z = 0; z < grid_dims.z; ++z) {
        positions[i] = grid_start + glm::vec3{
            static_cast<float>(x) * PARTICLE_SPACING,
            static_cast<float>(y) * PARTICLE_SPACING,
            static_cast<float>(z) * PARTICLE_SPACING
        };
        ++i;
    }
}

void FluidSimulator::reset() {
    init_positions();
}
