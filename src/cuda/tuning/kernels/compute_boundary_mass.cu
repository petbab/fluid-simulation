#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


__global__ void compute_boundary_mass(
    const float* positions, float* masses,
    unsigned fluid_n, unsigned boundary_n,
    const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= boundary_n)
        return;

    float4 xi = get_pos(positions, fluid_n + i);
    float sum = cubic_spline(0.f, SUPPORT_RADIUS);
    dev_n_search->for_neighbors(xi, [=, &sum] (unsigned j) {
        if (!is_boundary(j, fluid_n))
            return;

        float4 xj = get_pos(positions, j);
        if (is_neighbor(xi, xj, i, j)) {
            float4 r = xi - xj;
            float q = r_to_q(r, SUPPORT_RADIUS);
            sum += cubic_spline(q, SUPPORT_RADIUS);
        }
    });

    masses[i] = REST_DENSITY / sum;
}
