#include "simulator.h"


CUDASimulator::CUDASimulator(unsigned grid_count, const BoundingBox &bounding_box, bool is_2d)
    : FluidSimulator(grid_count, bounding_box, is_2d) {}

void CUDASimulator::init_buffer(GLuint vbo) {
    cuda_gl_buffer = std::make_unique<CUDAGLBuffer>(vbo);
}

void CUDASimulator::update(float delta) {
    if (cuda_gl_buffer == nullptr)
        return;

    CUDAGLBuffer::CUDALock lock = cuda_gl_buffer->lock();
    cuda_update_positions(lock.get_ptr());
}
