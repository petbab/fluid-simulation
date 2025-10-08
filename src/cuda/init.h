#pragma once

#include <cuda_runtime.h>


inline void cuda_init() {
    int dev = 0;
    cudaSetDevice(dev);
    cudaCheckError();
}
