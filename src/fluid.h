#pragma once

#include "object.h"


class Fluid : public Object {
public:
    Fluid(unsigned grid_count, float gap, BoundingBox bounding_box);

    void update(double delta) override;

private:
    void initiate_particles(unsigned grid_count, float gap);

    std::vector<glm::vec3> positions, velocities;
    BoundingBox bounding_box;
};
