#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


static constexpr float SURFACE_TENSION_ALPHA = 0.15f;

__global__ void compute_surface_tension(
    const float4* positions, const float* densities,
    const float4* normals, float4* acceleration,
    unsigned n, const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float4 xi = positions[i];
    float4 ni = normals[i];
    float di = densities[i];
    float4 f{0.f};

    dev_n_search->for_neighbors(xi, [=, &f] (unsigned j) {
        float4 xj = positions[j];

        if (is_neighbor(xi, xj, i, j)) {
            float4 x_ij = xi - positions[j];
            float q = r_to_q(x_ij, SUPPORT_RADIUS);
            if (dot(x_ij, x_ij) > 1e-6)
                f += (PARTICLE_MASS * normalize(x_ij)
                    * cohesion(q, SUPPORT_RADIUS) + ni - normals[j]) / (di + densities[j]);
        }
    });

    acceleration[i] -= SURFACE_TENSION_ALPHA * 2.f * REST_DENSITY * f;
}
