#include "cuda_gl_buffer.h"

#include <debug.h>
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

void* CUDAGLBuffer::CUDALock::get_ptr() const {
    void* ptr;
    cudaGraphicsResourceGetMappedPointer(&ptr, nullptr, cuda_resource);
    cudaCheckError();
    return ptr;
}

CUDAGLBuffer::CUDAGLBuffer(unsigned gl_id, cudaGraphicsRegisterFlags flags) {
    cudaGraphicsGLRegisterBuffer(&cuda_resource, gl_id, flags);
    cudaCheckError();
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
