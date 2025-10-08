#include "cuda_gl_buffer.h"

#include "../debug.h"
#include <cuda_gl_interop.h>


CUDAGLBuffer::CUDALock::CUDALock(cudaGraphicsResource* cuda_res) : cuda_resource{cuda_res} {
    cudaGraphicsMapResources(1, &cuda_resource);
    cudaCheckError();
}

CUDAGLBuffer::CUDALock::~CUDALock() {
    cudaGraphicsUnmapResources(1, &cuda_resource);
    cudaCheckError();
    cudaDeviceSynchronize();
    cudaCheckError();
}

float* CUDAGLBuffer::CUDALock::get_ptr() const {
    float* ptr;
    cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&ptr), nullptr, cuda_resource);
    cudaCheckError();
    return ptr;
}

CUDAGLBuffer::CUDAGLBuffer(unsigned vbo) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glCheckError();

    // Register buffer with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_resource, vbo, cudaGraphicsMapFlagsNone);
    cudaCheckError();

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

CUDAGLBuffer::~CUDAGLBuffer() {
    if (cuda_resource != nullptr) {
        cudaGraphicsUnregisterResource(cuda_resource);
        cudaCheckError();
    }
}

CUDAGLBuffer::CUDALock CUDAGLBuffer::lock() const {
    return CUDALock{cuda_resource};
}
