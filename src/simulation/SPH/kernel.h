#pragma once

#include <array>
#include <glm/vec3.hpp>


/**
 * @brief Base class for tabulated SPH kernel functions.
 *
 * Pre-computes kernel values and gradients in lookup tables for
 * efficient evaluation during simulation.
 */
class Kernel {
    static constexpr unsigned SAMPLES = 1000;  ///< Number of table samples.
    static constexpr float Q_STEP = 1.f / (SAMPLES - 1);  ///< Sample spacing.

    using table_t = std::array<float, SAMPLES>;  ///< Lookup table type.

public:
    /**
     * @brief Constructs and populates the kernel tables.
     * @param support_radius Support radius.
     */
    explicit Kernel(float support_radius);
    virtual ~Kernel() = default;

    /**
     * @brief Evaluates the kernel W(r).
     * @param r Distance vector.
     * @return Kernel value.
     */
    float W(const glm::vec3 &r) const;

    /**
     * @brief Evaluates the kernel gradient ∇W(r).
     * @param r Distance vector.
     * @return Gradient vector.
     */
    glm::vec3 grad_W(const glm::vec3 &r) const;

protected:
    /** @brief Fills the lookup tables. */
    void populate_tables();

    /**
     * @brief Computes W(q) for a given normalized distance.
     * @param q Normalized distance (|r| / h).
     * @return Kernel value.
     */
    virtual float compute_W(float q) const = 0;

    /**
     * @brief Computes |∇W(q)| for a given normalized distance.
     * @param q Normalized distance.
     * @return Gradient magnitude.
     */
    virtual float compute_grad_W(float q) const = 0;

private:
    static float sample(const table_t &t, float q);  ///< Table lookup with interpolation.

    const float inv_support_radius;  ///< 1 / support_radius.
    table_t table, grad_table;       ///< Precomputed W and |∇W| tables.
};

/**
 * @brief Cubic spline kernel (Monaghan 1992).
 */
class CubicSpline final : public Kernel {
public:
    /**
     * @brief Constructs the cubic spline kernel.
     * @param support_radius Support radius.
     */
    explicit CubicSpline(float support_radius);

protected:
    float compute_W(float q) const override;
    float compute_grad_W(float q) const override;

private:
    const float h, factor, grad_factor;  ///< Precomputed constants.
};

/**
 * @brief Spiky kernel (Muller et al. 2003).
 */
class SpikyKernel final : public Kernel {
public:
    /**
     * @brief Constructs the spiky kernel.
     * @param support_radius Support radius.
     */
    explicit SpikyKernel(float support_radius);

protected:
    float compute_W(float q) const override;
    float compute_grad_W(float q) const override;

private:
    const float h, factor, grad_factor;  ///< Precomputed constants.
};

/**
 * @brief Cohesion kernel for surface tension (Akinci et al. 2013).
 */
class CohesionKernel final : public Kernel {
public:
    /**
     * @brief Constructs the cohesion kernel.
     * @param support_radius Support radius.
     */
    explicit CohesionKernel(float support_radius);

protected:
    float compute_W(float q) const override;
    float compute_grad_W(float q) const override;

private:
    const float h, factor;  ///< Precomputed constants.
};
