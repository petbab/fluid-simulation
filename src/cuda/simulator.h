#pragma once

#include <glad/glad.h>
#include "cuda_gl_buffer.h"
#include "../simulation/fluid_simulator.h"


class CUDASimulator : public FluidSimulator {
public:
    CUDASimulator(unsigned grid_count, const BoundingBox &bounding_box, bool is_2d);

    void init_buffer(GLuint vbo);

protected:
    std::unique_ptr<CUDAGLBuffer> cuda_gl_positions;
};
