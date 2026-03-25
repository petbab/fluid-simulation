#define KERNEL_DIR /home/pbabic/Repositories/fluid-simulation/src/cuda/tuning/kernels
#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


__global__ void update_velocities(float4* velocities, const float4* acceleration, unsigned n, float delta) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    velocities[i] += acceleration[i] * delta;
}
