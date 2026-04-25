#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


static constexpr float VISCOSITY = 0.001f;

__global__ void compute_viscosity(
    const float4* positions, const float4* velocities,
    const float* densities, float4* acceleration,
    unsigned n, const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float4 velocity_laplacian{0.f};
    float4 xi = positions[i];
    float4 vi = velocities[i];

    dev_n_search->for_neighbors(xi, [=, &velocity_laplacian] (unsigned j) {
        if (is_boundary(j, n))
            return;

        float4 xj = positions[j];

        if (is_neighbor(xi, xj, i, j)) {
            float4 x_ij = xi - xj;
            float4 v_ij = vi - velocities[j];
            float q = r_to_q(x_ij, SUPPORT_RADIUS);

            velocity_laplacian += dot(v_ij, x_ij) * cubic_spline_grad(q, SUPPORT_RADIUS)
                * x_ij / (densities[j] * (dot(x_ij, x_ij)
                    + 0.01f * SUPPORT_RADIUS * SUPPORT_RADIUS));
        }
    });

    velocity_laplacian *= 10 * PARTICLE_MASS;
    acceleration[i] += VISCOSITY * velocity_laplacian;
}
