#pragma once

#include "sph_base.h"
#include <vector>


class DFSPHSimulator final : public SPHBase {
public:
    ///////////////////////////////////////////////////////////////////////////////
    ////                         SIMULATION PARAMETERS                         ////
    ///////////////////////////////////////////////////////////////////////////////
    static constexpr float ALPHA_DENOM_EPSILON = 1.e-5f;

    static constexpr float MAX_TIME_STEP = 0.001f;
    static constexpr float MIN_TIME_STEP = 0.0001f;

    static constexpr float MAX_DIVERGENCE_ERROR = 0.1f; // 0.001 * REST_DENSITY
    static constexpr float MAX_DENSITY_ERROR = 0.001f; // 0.001 * REST_DENSITY
    static constexpr int MAX_DIVERGENCE_ITERATIONS = 100;
    static constexpr int MAX_DENSITY_ITERATIONS = 100;
    ///////////////////////////////////////////////////////////////////////////////

    DFSPHSimulator(unsigned grid_count, const BoundingBox &bounding_box, bool is_2d = false);

    void update(float delta) override;

    void reset() override;

private:
    void compute_alphas();

    void correct_density_error(float delta);
    void correct_divergence_error(float delta);
    void warm_start_density(float delta);
    void warm_start_divergence(float delta);

    std::vector<float> predicted_densities,
        alphas, divergence_errors,
        divergence_kappas, density_kappas;

    CubicSpline kernel;

    bool first_iteration = true;
};
