#pragma once

#include "../cuda/particle_data_visualizer.cuh"
#include "../render/object.h"


class FluidSimulator {
public:
    ///////////////////////////////////////////////////////////////////////////////
    ////                         SIMULATION PARAMETERS                         ////
    ///////////////////////////////////////////////////////////////////////////////
    static constexpr float PARTICLE_RADIUS = 0.02f;
    static constexpr float PARTICLE_SPACING = 2.f * PARTICLE_RADIUS;
    ///////////////////////////////////////////////////////////////////////////////

    struct grid_dims_t {
        unsigned x, y, z;
    };

    struct opts_t {
        glm::vec3 origin;
        grid_dims_t grid_dims;
        const BoundingBox &bounding_box;
    };

    FluidSimulator(const opts_t &opts);
    virtual ~FluidSimulator() = default;

    virtual void update(float delta) = 0;

    auto get_position_data() -> std::span<const float>;

    virtual void reset();

    virtual void visualize(Shader *shader) {}

private:
    void init_positions();

protected:
    std::vector<glm::vec3> positions;
    const unsigned particle_count;

    const BoundingBox &bounding_box;
    const grid_dims_t grid_dims;
    const glm::vec3 origin;

    ParticleDataVisualizer visualizer;
};
