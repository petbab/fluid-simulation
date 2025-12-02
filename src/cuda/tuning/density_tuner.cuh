#pragma once

#include <Ktt.h>
#include <memory>
#include <filesystem>


class DensityTuner final {
public:
    explicit DensityTuner(unsigned particles);

    void run(float *positions_dev_ptr, float* densities_dev_ptr, unsigned particles);

private:
    std::unique_ptr<ktt::Tuner> tuner;
    ktt::KernelDefinitionId definition;
    ktt::KernelId kernel;
};
