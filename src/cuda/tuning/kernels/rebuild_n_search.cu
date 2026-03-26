#define KERNEL_DIR /home/pbabic/Repositories/fluid-simulation/src/cuda/tuning/kernels
#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


__global__ void rebuild_n_search(NSearch *dev_n_search, const float *particle_positions, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 pos = get_pos(particle_positions, i);
    dev_n_search->insert(pos, i);
}
