#pragma once

#include <glad/glad.h>
#include "cuda_gl_buffer.h"
#include <simulation/fluid_simulator.h>


class CUDASimulator : public FluidSimulator {
public:
    using FluidSimulator::FluidSimulator;

    void init_buffer(GLuint vbo) {
        cuda_gl_positions = std::make_unique<CUDAGLBuffer>(vbo);
    }

protected:
    std::unique_ptr<CUDAGLBuffer> cuda_gl_positions;
};
