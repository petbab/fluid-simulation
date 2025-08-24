#pragma once

#include "../render/object.h"

class FluidSimulator {
public:
    static constexpr float PARTICLE_RADIUS = 0.02f;
    static constexpr float PARTICLE_SPACING = 2.f * PARTICLE_RADIUS;

    FluidSimulator(unsigned grid_count, BoundingBox bounding_box, bool is_2d = false);
    virtual ~FluidSimulator() = default;

    virtual void update(double delta) = 0;

    auto get_position_data() -> std::span<const float>;

    virtual void reset();

private:
    void init_positions();

protected:
    std::vector<glm::vec3> positions;
    BoundingBox bounding_box;
    const unsigned grid_count;
    const bool is_2d;
};
