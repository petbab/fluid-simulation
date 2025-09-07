#pragma once

#include <array>
#include <glm/vec3.hpp>


class Kernel {
    static constexpr unsigned SAMPLES = 1000;
    static constexpr float Q_STEP = 1.f / (SAMPLES - 1);

    using table_t = std::array<float, SAMPLES>;

public:
    explicit Kernel(float support_radius);
    virtual ~Kernel() = default;

    float W(const glm::vec3 &r) const;
    glm::vec3 grad_W(const glm::vec3 &r) const;

protected:
    void populate_tables();

    virtual float compute_W(float q) const = 0;
    virtual float compute_grad_W(float q) const = 0;

private:
    static float sample(const table_t &t, float q);

    const float inv_support_radius;
    table_t table, grad_table;
};

class CubicSpline final : public Kernel {
public:
    explicit CubicSpline(float support_radius, bool is_2d = false);

protected:
    float compute_W(float q) const override;
    float compute_grad_W(float q) const override;

private:
    const float h, factor, grad_factor;
};

class SpikyKernel final : public Kernel {
public:
    explicit SpikyKernel(float support_radius);

protected:
    float compute_W(float q) const override;
    float compute_grad_W(float q) const override;

private:
    const float h, factor, grad_factor;
};

class CohesionKernel final : public Kernel {
public:
    explicit CohesionKernel(float support_radius);

protected:
    float compute_W(float q) const override;
    float compute_grad_W(float q) const override;

private:
    const float h, factor;
};
