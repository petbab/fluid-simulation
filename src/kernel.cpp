#include <glm/geometric.hpp>
#include <glm/ext/scalar_constants.hpp>
#include "kernel.h"


static float cubic_spline(const float q, const float factor) {
    if (q <= 0.5f)
        return factor * (6.f*q*q*q - 6.f*q*q + 1);
    if (q <= 1.f)
        return factor * 2.f * std::pow(1.f - q, 3.f);
    return 0.f;
}

static float cubic_spline_g(const float q, const float h, const float grad_factor) {
    if (q < 1.e-9 || q > 1.)
        return 0.f;

    float r_len = q * h;

    if (q <= 0.5)
        return grad_factor * q * (3.f*q - 2.f) / (h * r_len);
    return -grad_factor * (1.f - q) * (1.f - q) / (h * r_len);
}

CubicSpline::CubicSpline(float support_radius)
    : inv_support_radius{1.f / support_radius},
      factor{8.f / (glm::pi<float>() * support_radius * support_radius * support_radius)},
      grad_factor{48.f / (glm::pi<float>() * support_radius * support_radius * support_radius)},
      table{}, grad_table{} {
    for (unsigned i = 0; i < SAMPLES; ++i) {
        const float q = static_cast<float>(i) * Q_STEP;
        table[i] = cubic_spline(q, factor);
        grad_table[i] = cubic_spline_g(q, support_radius, grad_factor);
    }
}

float CubicSpline::W(const glm::vec3 &r) const {
    float q = glm::length(r) * inv_support_radius;
    if (q > 1.f)
        return 0;
    return sample(table, q);
}

glm::vec3 CubicSpline::grad_W(const glm::vec3 &r) const {
    float q = glm::length(r) * inv_support_radius;
    if (q > 1.f || q < 1.e-9f)
        return glm::vec3{0};
    return sample(grad_table, q) * r;
}

float CubicSpline::sample(const std::array<float, SAMPLES> &t, float q) {
    const float index_f = q * (SAMPLES - 1);
    const auto lo = static_cast<unsigned>(index_f);

    if (lo >= SAMPLES - 1)
        return t[SAMPLES - 1];

    const float lambda = index_f - static_cast<float>(lo);
    return t[lo] + lambda * (t[lo + 1] - t[lo]);
}
