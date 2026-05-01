#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


__global__ void compute_boundary_mass(
    const float4* positions, float* masses, unsigned boundary_n,
    const NSearch *boundary_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= boundary_n)
        return;

    float4 xi = positions[i];
    float sum = cubic_spline(0.f, SUPPORT_RADIUS);
    boundary_n_search->for_neighbors(xi, [=, &sum] (unsigned j) {
        float4 xj = positions[j];
        if (is_neighbor(xi, xj, i, j)) {
            float4 r = xi - xj;
            float q = r_to_q(r, SUPPORT_RADIUS);
            sum += cubic_spline(q, SUPPORT_RADIUS);
        }
    });

    masses[i] = REST_DENSITY / sum;
}
