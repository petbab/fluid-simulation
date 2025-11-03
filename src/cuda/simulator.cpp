#include "simulator.h"


CUDASimulator::CUDASimulator(unsigned grid_count, const BoundingBox &bounding_box, bool is_2d)
    : FluidSimulator(grid_count, bounding_box, is_2d) {}

void CUDASimulator::init_buffer(GLuint vbo) {
    cuda_gl_positions = std::make_unique<CUDAGLBuffer>(vbo);
}
