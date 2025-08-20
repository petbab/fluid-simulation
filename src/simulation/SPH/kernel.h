#pragma once

#include <array>
#include <glm/vec3.hpp>


template<class K>
concept Kernel = requires(K k) {
    { k.W(glm::vec3{0.}) } -> std::same_as<float>;
    { k.grad_W(glm::vec3{0.}) } -> std::same_as<glm::vec3>;
};

class CubicSpline {
    static constexpr unsigned SAMPLES = 1000;
    static constexpr float Q_STEP = 1.f / (SAMPLES - 1);

public:
    explicit CubicSpline(float support_radius, bool is_2d = false);

    float W(const glm::vec3 &r) const;
    glm::vec3 grad_W(const glm::vec3 &r) const;

private:
    static float sample(const std::array<float, SAMPLES> &t, float q);

    const float inv_support_radius, factor, grad_factor;
    std::array<float, SAMPLES> table, grad_table;
};

static_assert(Kernel<CubicSpline>);
