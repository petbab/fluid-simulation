#ifdef NOT_IN_KTT
#include "common.cuh"
#endif

#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


__global__ void count_neighbors(
    const float4* positions,
    unsigned* counts,
    unsigned n,
    const NSearch* dev_n_search)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 xi = positions[i];
    unsigned c = 0;
    dev_n_search->for_neighbors<count_neighbors_u_n>(xi, [=, &c](unsigned j) {
        if (is_neighbor(xi, positions[j], i, j))
            ++c;
    });
    counts[i] = c;
}

__global__ void fill_neighbors(
    const float4* positions,
    const unsigned* offsets,
    unsigned* neighbors,
    unsigned n,
    const NSearch* dev_n_search)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 xi = positions[i];
    unsigned w = offsets[i];
    dev_n_search->for_neighbors<fill_neighbors_u_n>(xi, [=, &w](unsigned j) {
        if (is_neighbor(xi, positions[j], i, j))
            neighbors[w++] = j;
    });
}
