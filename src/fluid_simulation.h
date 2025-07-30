#pragma once

#include <vector>
#include <glm/vec3.hpp>
#include <span>
#include "object.h"


class FluidSimulation {
public:
    FluidSimulation(unsigned grid_count, float gap, BoundingBox bounding_box);

    void update(double delta);

    auto get_positions() -> std::span<const float>;

private:
    void initiate_particles(unsigned grid_count, float gap);

    std::vector<glm::vec3> positions, velocities;
    BoundingBox bounding_box;
};
