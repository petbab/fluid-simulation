#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


static constexpr float STIFFNESS = 0.1f;
static constexpr float EXPONENT = 7.f;

__global__ void compute_pressure(const float* densities, float* pressures, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float d = fmaxf(densities[i], REST_DENSITY);
    pressures[i] = STIFFNESS *
        (powf(d / REST_DENSITY, EXPONENT) - 1.f);
}
