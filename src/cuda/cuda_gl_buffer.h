#pragma once

#include <cuda_runtime.h>


/**
 * @brief RAII wrapper for a CUDA-registered OpenGL buffer.
 *
 * Maps an existing OpenGL buffer object into CUDA address space
 * so kernels can write directly into GPU vertex buffers.
 */
class CUDAGLBuffer {
public:
    /**
     * @brief RAII lock for CUDA graphics resource mapping.
     *
     * On construction the resource is mapped into CUDA address space;
     * on destruction it is unmapped.
     */
    struct CUDALock {
        /**
         * @brief Maps the graphics resource.
         * @param cuda_resource CUDA graphics resource to lock.
         */
        explicit CUDALock(cudaGraphicsResource* cuda_resource);
        CUDALock(const CUDALock &) = delete;
        CUDALock &operator=(const CUDALock &) = delete;
        CUDALock(CUDALock &&) noexcept;
        CUDALock &operator=(CUDALock &&) noexcept;
        ~CUDALock();

        /** @return Device pointer to the mapped buffer data. */
        void* get_ptr() const;

    private:
        cudaGraphicsResource* cuda_resource = nullptr;  ///< Underlying CUDA graphics resource.
    };

    /**
     * @brief Registers an existing OpenGL buffer with CUDA.
     * @param gl_id OpenGL buffer identifier.
     * @param flags CUDA registration flags.
     */
    CUDAGLBuffer(unsigned gl_id, cudaGraphicsRegisterFlags flags = cudaGraphicsRegisterFlagsNone);
    CUDAGLBuffer(const CUDAGLBuffer &) = delete;
    CUDAGLBuffer &operator=(const CUDAGLBuffer &) = delete;
    ~CUDAGLBuffer();

    /** @brief Locks the buffer for CUDA access. */
    CUDALock lock() const;

private:
    cudaGraphicsResource* cuda_resource = nullptr;  ///< CUDA graphics resource handle.
};
