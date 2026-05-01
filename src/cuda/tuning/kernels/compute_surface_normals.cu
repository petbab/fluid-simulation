#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


__global__ void compute_surface_normals(
    const float4* positions, const float* densities,
    float4* normals, unsigned n,
    const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float4 xi = positions[i];
    float4 normal{0.f};

    dev_n_search->for_neighbors(xi, [=, &normal] (unsigned j) {
        float4 xj = positions[j];

        if (is_neighbor(xi, xj, i, j)) {
            float4 r = xi - xj;
            float q = r_to_q(r, SUPPORT_RADIUS);
            normal += r * cubic_spline_grad(q, SUPPORT_RADIUS) / densities[j];
        }
    });

    normals[i] = SUPPORT_RADIUS * PARTICLE_MASS * normal;
}
