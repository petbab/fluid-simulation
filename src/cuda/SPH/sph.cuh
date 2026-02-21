#pragma once
#include "sph_base.cuh"


class CUDASPHSimulator final : public CUDASPHBase {
public:
    ///////////////////////////////////////////////////////////////////////////////
    ////                         SIMULATION PARAMETERS                         ////
    ///////////////////////////////////////////////////////////////////////////////
    // static constexpr float STIFFNESS = 1.f;
    // static constexpr float EXPONENT = 3.f;
    static constexpr float STIFFNESS = 0.1f;
    static constexpr float EXPONENT = 7.f;

    static constexpr float MAX_TIME_STEP = 0.0005f;
    static constexpr float MIN_TIME_STEP = 0.00001f;
    ///////////////////////////////////////////////////////////////////////////////

    CUDASPHSimulator(const opts_t &opts);
    void update(float delta) override;

    void visualize(Shader* shader) override;

    void reset() override;

private:
    void compute_pressure();
    void apply_pressure_force(const float *positions_dev_ptr, float delta);

    thrust::device_vector<float> pressure;
};
