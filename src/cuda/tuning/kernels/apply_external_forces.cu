#define KERNEL_DIR /home/pbabic/Repositories/fluid-simulation/src/cuda/tuning/kernels
#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)

#ifndef EXTERNAL_FORCE
#define EXTERNAL_FORCE ([](float4 pos) { return make_float4(0., 0., 0., 0.); })
#endif


static constexpr float4 GRAVITY{0, -9.81f, 0, 0};

__global__ void apply_external_forces(const float* positions, float4* acceleration, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        acceleration[i] = GRAVITY + EXTERNAL_FORCE(get_pos(positions, i));
}
