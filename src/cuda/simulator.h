#pragma once

#include <glad/glad.h>
#include "cuda_gl_buffer.h"
#include "../simulation/fluid_simulator.h"


class CUDASimulator : public FluidSimulator {
public:
    CUDASimulator(grid_dims_t grid_dims, const BoundingBox &bounding_box,
        const std::vector<const Object*> &collision_objects);

    void init_buffer(GLuint vbo);

protected:
    std::unique_ptr<CUDAGLBuffer> cuda_gl_positions;
};
