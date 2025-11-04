#include "sph.cuh"

#include "kernel.cuh"
#include "../../debug.h"

///////////////////////////////////////////////////////////////////////////////
////                        KERNEL HYPERPARAMETERS                         ////
///////////////////////////////////////////////////////////////////////////////

static constexpr dim3 COMPUTE_PRESSURE_BLOCK_SIZE = {128};
static constexpr dim3 APPLY_PRESSURE_FORCE_BLOCK_SIZE = {128};

///////////////////////////////////////////////////////////////////////////////
////                                KERNELS                                ////
///////////////////////////////////////////////////////////////////////////////

__global__ void compute_pressure_k(const float* densities, float* pressures, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float d = fmaxf(densities[i], CUDASPHSimulator::REST_DENSITY);
    pressures[i] = CUDASPHSimulator::STIFFNESS *
        (powf(d / CUDASPHSimulator::REST_DENSITY, CUDASPHSimulator::EXPONENT) - 1.f);
}

__global__ void apply_pressure_force_k(const float* positions, const float* densities,
                                       const float* pressures, glm::vec3* velocities,
                                       unsigned n, float delta) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    glm::vec3 p_accel{0.f};
    glm::vec3 xi = get_pos(positions, i);
    float di = densities[i];
    float dpi = pressures[i] / (di * di);

    for (unsigned j = 0; j < n; ++j) {
        glm::vec3 xj = get_pos(positions, j);
        if (!is_neighbor(xi, xj, i, j))
            continue;

        float dj = densities[j];
        float dpj = pressures[j] / (dj * dj);

        glm::vec3 r = xi - xj;
        float q = r_to_q(r, CUDASPHSimulator::SUPPORT_RADIUS);
        p_accel -= (dpi + dpj) * spiky_grad(q, CUDASPHSimulator::SUPPORT_RADIUS) * r;
    }

    velocities[i] += delta * CUDASPHSimulator::PARTICLE_MASS * p_accel;
}

///////////////////////////////////////////////////////////////////////////////
////                           CUDASPHSimulator                            ////
///////////////////////////////////////////////////////////////////////////////

CUDASPHSimulator::CUDASPHSimulator(unsigned grid_count, const BoundingBox& bounding_box, bool is_2d)
    : CUDASPHBase(grid_count, bounding_box, is_2d), pressure(particle_count) {
}

void CUDASPHSimulator::update(float delta) {
    auto lock = cuda_gl_positions->lock();
    float* positions_ptr = lock.get_ptr();

    compute_densities(positions_ptr);

    apply_non_pressure_forces(positions_ptr, delta);

    delta = std::clamp(delta, MIN_TIME_STEP, MAX_TIME_STEP);

    compute_pressure();
    apply_pressure_force(positions_ptr, delta);

    update_positions(positions_ptr, delta);
}

void CUDASPHSimulator::reset() {
    CUDASPHBase::reset();

    thrust::fill(pressure.begin(), pressure.end(), 0);
}

void CUDASPHSimulator::compute_pressure() {
    const float* densities_ptr = thrust::raw_pointer_cast(density.data());
    float* pressures_ptr = thrust::raw_pointer_cast(pressure.data());

    const dim3 grid_size{particle_count / COMPUTE_PRESSURE_BLOCK_SIZE.x + 1};
    compute_pressure_k<<<COMPUTE_PRESSURE_BLOCK_SIZE, grid_size>>>(
        densities_ptr, pressures_ptr, particle_count);
    cudaCheckError();
}

void CUDASPHSimulator::apply_pressure_force(const float* positions_dev_ptr, float delta) {
    const float* densities_ptr = thrust::raw_pointer_cast(density.data());
    const float* pressures_ptr = thrust::raw_pointer_cast(pressure.data());
    glm::vec3* velocities_ptr = thrust::raw_pointer_cast(velocity.data());

    const dim3 grid_size{particle_count / APPLY_PRESSURE_FORCE_BLOCK_SIZE.x + 1};
    apply_pressure_force_k<<<APPLY_PRESSURE_FORCE_BLOCK_SIZE, grid_size>>>(
        positions_dev_ptr, densities_ptr, pressures_ptr, velocities_ptr, particle_count, delta);
    cudaCheckError();
}
