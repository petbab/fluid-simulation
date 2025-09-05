#pragma once

#include "common.h"


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

    SPHSimulator(unsigned grid_count, const BoundingBox &bounding_box, bool is_2d = false);

    void update(double delta) override;

    void reset() override;

private:
    void compute_pressure();
    void apply_pressure_force(double delta);

    std::vector<float> pressure;
    SpikyKernel spiky_k;
};
