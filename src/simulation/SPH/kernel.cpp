#include <glm/geometric.hpp>
#include <glm/ext/scalar_constants.hpp>
#include "kernel.h"


Kernel::Kernel(float support_radius) : inv_support_radius{1.f / support_radius}, table{}, grad_table{} {}

float Kernel::W(const glm::vec3 &r) const {
    float q = glm::length(r) * inv_support_radius;
    if (q > 1.f)
        return 0;
    return sample(table, q);
}

glm::vec3 Kernel::grad_W(const glm::vec3 &r) const {
    float q = glm::length(r) * inv_support_radius;
    if (q > 1.f || q < 1.e-9f)
        return glm::vec3{0};
    return sample(grad_table, q) * r;
}

void Kernel::populate_tables() {
    for (unsigned i = 0; i < SAMPLES; ++i) {
        const float q = static_cast<float>(i) * Q_STEP;
        table[i] = compute_W(q);
        grad_table[i] = compute_grad_W(q);
    }
}

float Kernel::sample(const table_t &t, float q) {
    const float index_f = q * (SAMPLES - 1);
    const auto lo = static_cast<unsigned>(index_f);

    if (lo >= SAMPLES - 1)
        return t[SAMPLES - 1];

    const float lambda = index_f - static_cast<float>(lo);
    return t[lo] + lambda * (t[lo + 1] - t[lo]);
}

CubicSpline::CubicSpline(float support_radius, bool is_2d)
    : Kernel{support_radius}, h{support_radius},
      factor{is_2d
        ? 40.f / (7.f * glm::pi<float>() * support_radius * support_radius)
        : 8.f / (glm::pi<float>() * support_radius * support_radius * support_radius)
      },
      grad_factor{is_2d
        ? 240.f / (7.f * glm::pi<float>() * support_radius * support_radius)
        : 48.f / (glm::pi<float>() * support_radius * support_radius * support_radius)
      } {
    populate_tables();
}

float CubicSpline::compute_W(float q) const {
    if (q <= 0.5f)
        return factor * (6.f*q*q*q - 6.f*q*q + 1);
    if (q <= 1.f)
        return factor * 2.f * std::pow(1.f - q, 3.f);
    return 0.f;
}

float CubicSpline::compute_grad_W(float q) const {
    if (q < 1.e-9 || q > 1.)
        return 0.f;

    float r_len = q * h;

    if (q <= 0.5)
        return grad_factor * q * (3.f*q - 2.f) / (h * r_len);
    return -grad_factor * (1.f - q) * (1.f - q) / (h * r_len);
}

SpikyKernel::SpikyKernel(float support_radius)
    : Kernel{support_radius}, h{support_radius},
      factor{15.f / (glm::pi<float>() * std::pow(support_radius, 9.f))},
      grad_factor{-45.f / (glm::pi<float>() * std::pow(support_radius, 9.f))} {
    populate_tables();
}

float SpikyKernel::compute_W(float q) const {
    return (q <= 1.f) ? factor * std::pow(1 - q, 3.f) : 0.f;
}

float SpikyKernel::compute_grad_W(float q) const {
    if (q < 1.e-9 || q > 1.)
        return 0.f;

    float r_len = q * h;
    return grad_factor * (1.f - q) * (1.f - q) / (h * r_len);
}
