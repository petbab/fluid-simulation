#pragma once

#include "tuner.h"


class DensityTuner final : public Tuner {
public:
    explicit DensityTuner(unsigned particles);

    void run(float *positions_dev_ptr, float* densities_dev_ptr, unsigned particles);
};
