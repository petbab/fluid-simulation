#pragma once

#include "sph_base.h"


class SPHSimulator final : public SPHBase {
public:
    ///////////////////////////////////////////////////////////////////////////////
    ////                         SIMULATION PARAMETERS                         ////
    ///////////////////////////////////////////////////////////////////////////////
    static constexpr float STIFFNESS = 1.f;
    static constexpr float EXPONENT = 3.f;

    static constexpr float MAX_TIME_STEP = 0.0005f;
    static constexpr float MIN_TIME_STEP = 0.00001f;
    ///////////////////////////////////////////////////////////////////////////////

    SPHSimulator(const opts_t &opts);

    void update(float delta) override;

    void reset() override;

private:
    void compute_pressure();
    void apply_pressure_force(float delta);

    std::vector<float> pressure;
    SpikyKernel spiky_k;
};
