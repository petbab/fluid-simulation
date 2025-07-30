#pragma once

#include "object.h"
#include "fluid_simulation.h"


class Fluid : public Object {
public:
    Fluid(unsigned grid_count, float gap, BoundingBox bounding_box);

    void update(double delta) override;

private:
    FluidSimulation simulation;
};
