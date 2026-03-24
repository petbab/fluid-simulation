#pragma once

#include "tuner.h"


class DensityTuner final : public Tuner {
public:
    explicit DensityTuner(unsigned particles);

    void run(float *positions_dev_ptr, float* densities_dev_ptr,
        float* boundary_mass_dev_ptr, void *dev_n_search,
        unsigned total_particles, unsigned fluid_particles);
};
