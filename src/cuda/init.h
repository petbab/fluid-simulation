#pragma once

#include <cuda_runtime.h>


/**
 * @brief Initializes the CUDA runtime on device 0.
 *
 * Calls cudaSetDevice(0) and checks for errors.
 */
inline void cuda_init() {
    int dev = 0;
    cudaSetDevice(dev);
    cudaCheckError();
}
