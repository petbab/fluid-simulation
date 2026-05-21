#ifdef NOT_IN_KTT
#include "common.cuh"
#endif

#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


__global__ void compute_morton(const float4* position, morton_t* morton, unsigned n) {
    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    NSearch::cell_t c = NSearch::cell_coord(position[i]);
    morton[i] = encode_morton(c);
}
