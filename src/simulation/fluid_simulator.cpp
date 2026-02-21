#include "fluid_simulator.h"
#include "SPH/boundary.h"
#include <random>


template<int ELEM_SIZE>
static inline auto vec_to_span(const std::vector<glm::vec<ELEM_SIZE, float>> &v) -> std::span<const float> {
    return {reinterpret_cast<const float *>(v.data()), v.size() * ELEM_SIZE};
}

FluidSimulator::FluidSimulator(const opts_t &opts)
    : fluid_particles{opts.grid_dims.x * opts.grid_dims.y * opts.grid_dims.z},
      bounding_box{opts.bounding_box}, grid_dims{opts.grid_dims}, origin{opts.origin} {
    init_positions();
    init_boundary_particles(opts.collision_objects);

    total_particles = positions.size();
    boundary_particles = total_particles - fluid_particles;

    visualizer = std::make_unique<ParticleDataVisualizer>(total_particles, fluid_particles);
}

auto FluidSimulator::get_position_data() -> std::span<const float> {
    return vec_to_span(positions);
}

void FluidSimulator::init_positions() {
    assert(grid_dims.x > 1);
    assert(grid_dims.y > 1);
    assert(grid_dims.z > 1);

    const glm::vec3 grid_start = origin - glm::vec3{
            static_cast<float>(grid_dims.x - 1),
            static_cast<float>(grid_dims.y - 1),
            static_cast<float>(grid_dims.z - 1)
        } * PARTICLE_RADIUS;

    assert(grid_start.x >= bounding_box.min.x);
    assert(grid_start.y >= bounding_box.min.y);
    assert(grid_start.z >= bounding_box.min.z);

    if (positions.size() < fluid_particles)
        positions.resize(fluid_particles);
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

void FluidSimulator::init_boundary_particles(const std::vector<const Object*>& collision_objects) {
    std::vector<glm::vec3> triangle_vertices;
    for (auto obj : collision_objects) {
        obj->get_geometry()->load_triangles(triangle_vertices, obj->get_model());
        generate_boundary_particles(positions, triangle_vertices, PARTICLE_RADIUS);
    }
}

void FluidSimulator::reset() {
    init_positions();
}
