#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


static constexpr float STIFFNESS = 0.1f;
static constexpr float EXPONENT = 7.f;
static constexpr float MAX_DENSITY_RATIO = 1.5f;

__global__ void compute_pressure(const float* densities, float* pressures, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float ratio = fmaxf(densities[i] / REST_DENSITY, 1.0f);
    ratio = fminf(ratio, MAX_DENSITY_RATIO);
    pressures[i] = STIFFNESS * (powf(ratio, EXPONENT) - 1.f);
}
