#pragma once

#include <render/box.h>


/**
 * @brief Base class for fluid simulators.
 *
 * Manages particle initialization, boundary generation, and provides
 * the interface for simulation updates and visualization.
 */
class FluidSimulator {
public:
    ///////////////////////////////////////////////////////////////////////////////
    ////                         SIMULATION PARAMETERS                         //////
    ///////////////////////////////////////////////////////////////////////////////
    static constexpr float PARTICLE_RADIUS = 0.02f;     ///< Particle radius.
    static constexpr float PARTICLE_SPACING = 2.f * PARTICLE_RADIUS;  ///< Particle spacing.
    ///////////////////////////////////////////////////////////////////////////////

    /** @brief Grid dimensions for particle initialization. */
    struct grid_dims_t {
        /**
         * @brief Constructs grid dimensions.
         * @param x Grid cells in X.
         * @param y Grid cells in Y.
         * @param z Grid cells in Z.
         */
        grid_dims_t(unsigned x, unsigned y, unsigned z) : x(x), y(y), z(z) {}

        /**
         * @brief Constructs cubic grid dimensions.
         * @param x Grid cells in all axes.
         */
        grid_dims_t(unsigned x) : grid_dims_t{x, x, x} {}

        unsigned x, y, z;  ///< Grid dimensions.
    };

    /** @brief Simulation initialization options. */
    struct opts_t {
        glm::vec3 origin;                              ///< Origin of the fluid volume.
        grid_dims_t grid_dims;                         ///< Grid dimensions for particle placement.
        const BoundingBox &bounding_box;               ///< Simulation bounding box.
        const std::vector<const Object*> &collision_objects;  ///< Objects for boundary generation.
        std::string external_force;                    ///< Optional external force macro name.
    };

    /**
     * @brief Constructs the fluid simulator.
     * @param opts Simulation options.
     */
    FluidSimulator(const opts_t &opts);
    virtual ~FluidSimulator() = default;

    /**
     * @brief Advances the simulation by one step.
     * @param delta Time step in seconds.
     */
    virtual void update(float delta) = 0;

    /** @return Span of current particle position data. */
    auto get_position_data() -> std::span<const float>;

    /** @brief Resets the simulation to initial state. */
    virtual void reset();

    /** @return Number of fluid particles. */
    unsigned get_fluid_particles() const { return fluid_particles; }

    /**
     * @brief Configures shader uniforms for visualization.
     * @param shader Shader to configure.
     */
    virtual void visualize(Shader *shader) {}

private:
    void init_positions();
    void init_boundary_particles(const std::vector<const Object*> &collision_objects);

protected:
    std::vector<glm::vec4> positions;  ///< CPU-side particle positions.
    unsigned total_particles, fluid_particles, boundary_particles;  ///< Particle counts.

    const BoundingBox &bounding_box;   ///< Simulation bounding box.
    const grid_dims_t grid_dims;       ///< Grid dimensions.
    const glm::vec3 origin;            ///< Fluid origin.
};
