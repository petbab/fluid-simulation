#include "simulator.h"


CUDASimulator::CUDASimulator(grid_dims_t grid_dims, const BoundingBox &bounding_box,
    const std::vector<const Object*> &collision_objects)
    : FluidSimulator(grid_dims, bounding_box, collision_objects) {}

void CUDASimulator::init_buffer(GLuint vbo) {
    cuda_gl_positions = std::make_unique<CUDAGLBuffer>(vbo);
}
