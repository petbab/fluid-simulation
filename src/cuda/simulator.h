#pragma once

#include <glad/glad.h>
#include "cuda_gl_buffer.h"
#include "../simulation/fluid_simulator.h"


class CUDASimulator final : public FluidSimulator {
public:
    CUDASimulator(unsigned grid_count, const BoundingBox &bounding_box, bool is_2d);

    void init_buffer(GLuint vbo);

    void update(float delta) override;

private:
    void cuda_update_positions(float *positions_ptr) const;

    std::unique_ptr<CUDAGLBuffer> cuda_gl_buffer;
};
