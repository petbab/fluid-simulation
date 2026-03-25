#pragma once

#include "tuner.h"
#include <render/box.h>


class UpdatePositionsTuner final : public Tuner {
public:
    explicit UpdatePositionsTuner(unsigned fluid_particles);

    void run(float *positions_dev_ptr, float4* velocities_dev_ptr,
        unsigned n, float delta, const BoundingBox &bb);
};
