#pragma once

#include "object.h"

class FluidSimulator {
public:
    static constexpr float PARTICLE_RADIUS = 0.02f;
    static constexpr float PARTICLE_SPACING = 2.f * PARTICLE_RADIUS;

    FluidSimulator(unsigned grid_count, BoundingBox bounding_box);
    virtual ~FluidSimulator() = default;

    virtual void update(double delta) = 0;

    auto get_position_data() -> std::span<const float>;

private:
    void init_positions(unsigned grid_count);

protected:
    std::vector<glm::vec3> positions;
    BoundingBox bounding_box;
};
