#include "simulator.h"


CUDASimulator::CUDASimulator(grid_dims_t grid_dims, const BoundingBox &bounding_box)
    : FluidSimulator(grid_dims, bounding_box) {}

void CUDASimulator::init_buffer(GLuint vbo) {
    cuda_gl_positions = std::make_unique<CUDAGLBuffer>(vbo);
}
