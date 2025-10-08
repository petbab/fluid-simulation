#pragma once

#include <cuda_runtime.h>


class CUDAGLBuffer {
public:
    struct CUDALock {
        explicit CUDALock(cudaGraphicsResource* cuda_resource);
        CUDALock(const CUDALock &) = delete;
        CUDALock &operator=(const CUDALock &) = delete;
        CUDALock(CUDALock &&) = delete;
        CUDALock &operator=(CUDALock &&) = delete;
        ~CUDALock();

        float* get_ptr() const;

    private:
        cudaGraphicsResource* cuda_resource = nullptr;
    };

    explicit CUDAGLBuffer(unsigned vbo);
    CUDAGLBuffer(const CUDAGLBuffer &) = delete;
    CUDAGLBuffer &operator=(const CUDAGLBuffer &) = delete;
    ~CUDAGLBuffer();

    CUDALock lock() const;

private:
    cudaGraphicsResource* cuda_resource = nullptr;
};
