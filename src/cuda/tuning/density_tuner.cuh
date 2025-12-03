#pragma once

#include <Ktt.h>
#include <memory>


class DensityTuner final {
public:
    explicit DensityTuner(unsigned particles);
    ~DensityTuner();

    void run(float *positions_dev_ptr, float* densities_dev_ptr, unsigned particles);

private:
    void print_best_config() const;

    std::unique_ptr<ktt::Tuner> tuner;
    ktt::KernelDefinitionId definition;
    ktt::KernelId kernel;
};
