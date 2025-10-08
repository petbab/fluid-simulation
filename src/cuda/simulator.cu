#include "simulator.h"


__global__ void update_positions_k(float *positions, unsigned size) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        positions[i] += 0.1f;
}

void CUDASimulator::cuda_update_positions(float *positions_ptr) const {
    unsigned threads = 256;
    unsigned size = positions.size() * 3;
    unsigned blocks = (size + threads - 1) / threads;
    update_positions_k<<<blocks, threads>>>(positions_ptr, size);
}
