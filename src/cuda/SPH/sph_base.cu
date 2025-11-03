#include "sph_base.cuh"
#include "../../debug.h"
#include "kernel.cuh"


///////////////////////////////////////////////////////////////////////////////
////                        KERNEL HYPERPARAMETERS                         ////
///////////////////////////////////////////////////////////////////////////////

static constexpr dim3 COMPUTE_DENSITY_BLOCK_SIZE = {128};
static constexpr dim3 UPDATE_POSITIONS_BLOCK_SIZE = {128};
static constexpr dim3 RESOLVE_COLLISIONS_BLOCK_SIZE = {128};
static constexpr dim3 APPLY_NON_PRESSURE_FORCES_BLOCK_SIZE = {128};

///////////////////////////////////////////////////////////////////////////////
////                                KERNELS                                ////
///////////////////////////////////////////////////////////////////////////////

__global__ void compute_densities_k(const float *positions, float *densities, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    glm::vec3 xi = get_pos(positions, i);
    float density = 0.f;
    for (unsigned j = 0; j < n; ++j) {
        glm::vec3 xj = get_pos(positions, j);
        float q = r_to_q(xi - xj, CUDASPHBase::SUPPORT_RADIUS);
        density += cubic_spline(q, CUDASPHBase::SUPPORT_RADIUS);
    }
    densities[i] = density * CUDASPHBase::PARTICLE_MASS;
}

__global__ void update_positions_k(float* positions, const glm::vec3* velocities, unsigned n, float delta) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    glm::vec3 pos = get_pos(positions, i);
    pos += delta * velocities[i];
    set_pos(positions, i, pos);
}

__global__ void resolve_collisions_k(float* positions, glm::vec3* velocities, unsigned n, BoundingBox bb) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    glm::vec3 pos = get_pos(positions, i);
    glm::vec3 pos_begin = pos;
    glm::vec3 vel = velocities[i];
    if (pos.x - CUDASPHBase::PARTICLE_RADIUS < bb.min.x) {
        pos.x = bb.min.x + CUDASPHBase::PARTICLE_RADIUS;
        vel.x *= -CUDASPHBase::ELASTICITY;
    } else if (pos.x + CUDASPHBase::PARTICLE_RADIUS > bb.max.x) {
        pos.x = bb.max.x - CUDASPHBase::PARTICLE_RADIUS;
        vel.x *= -CUDASPHBase::ELASTICITY;
    }
    if (pos.y - CUDASPHBase::PARTICLE_RADIUS < bb.min.y) {
        pos.y = bb.min.y + CUDASPHBase::PARTICLE_RADIUS;
        vel.y *= -CUDASPHBase::ELASTICITY;
    } else if (pos.y + CUDASPHBase::PARTICLE_RADIUS > bb.max.y) {
        pos.y = bb.max.y - CUDASPHBase::PARTICLE_RADIUS;
        vel.y *= -CUDASPHBase::ELASTICITY;
    }
    if (pos.z - CUDASPHBase::PARTICLE_RADIUS < bb.min.z) {
        pos.z = bb.min.z + CUDASPHBase::PARTICLE_RADIUS;
        vel.z *= -CUDASPHBase::ELASTICITY;
    } else if (pos.z + CUDASPHBase::PARTICLE_RADIUS > bb.max.z) {
        pos.z = bb.max.z - CUDASPHBase::PARTICLE_RADIUS;
        vel.z *= -CUDASPHBase::ELASTICITY;
    }

    if (pos_begin != pos) {
        set_pos(positions, i, pos);
        velocities[i] = vel;
    }
}

__global__ void apply_non_pressure_forces_k(glm::vec3* non_pressure_accel, glm::vec3* velocities, unsigned n, float delta) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    // '*' takes the vector by reference
    const glm::vec3 g = CUDASPHBase::GRAVITY;
    velocities[i] += delta * g;
}

///////////////////////////////////////////////////////////////////////////////
////                              CUDASPHBase                              ////
///////////////////////////////////////////////////////////////////////////////

CUDASPHBase::CUDASPHBase(unsigned grid_count, const BoundingBox &bounding_box, bool is_2d)
    : CUDASimulator(grid_count, bounding_box, is_2d),
      density(particle_count),
      velocity(particle_count),
      non_pressure_accel(particle_count) {}

void CUDASPHBase::compute_densities(const float *positions_dev_ptr) {
    float *densities_ptr = thrust::raw_pointer_cast(density.data());

    const dim3 grid_size{particle_count / COMPUTE_DENSITY_BLOCK_SIZE.x + 1};
    compute_densities_k<<<COMPUTE_DENSITY_BLOCK_SIZE, grid_size>>>(
        positions_dev_ptr, densities_ptr, particle_count);
    cudaCheckError();
}

void CUDASPHBase::update_positions(float *positions_dev_ptr, float delta) {
    glm::vec3 *velocities_ptr = thrust::raw_pointer_cast(velocity.data());

    const dim3 grid_size{particle_count / UPDATE_POSITIONS_BLOCK_SIZE.x + 1};
    update_positions_k<<<UPDATE_POSITIONS_BLOCK_SIZE, grid_size>>>(
        positions_dev_ptr, velocities_ptr, particle_count, delta);
    cudaCheckError();
}

void CUDASPHBase::resolve_collisions(float *positions_dev_ptr) {
    glm::vec3 *velocities_ptr = thrust::raw_pointer_cast(velocity.data());

    const dim3 grid_size{particle_count / RESOLVE_COLLISIONS_BLOCK_SIZE.x + 1};
    resolve_collisions_k<<<RESOLVE_COLLISIONS_BLOCK_SIZE, grid_size>>>(
        positions_dev_ptr, velocities_ptr, particle_count, bounding_box);
    cudaCheckError();
}

void CUDASPHBase::apply_non_pressure_forces(float delta) {
    glm::vec3 *velocities_ptr = thrust::raw_pointer_cast(velocity.data());
    glm::vec3 *non_pressure_accel_ptr = thrust::raw_pointer_cast(non_pressure_accel.data());

    const dim3 grid_size{particle_count / APPLY_NON_PRESSURE_FORCES_BLOCK_SIZE.x + 1};
    apply_non_pressure_forces_k<<<APPLY_NON_PRESSURE_FORCES_BLOCK_SIZE, grid_size>>>(
        non_pressure_accel_ptr, velocities_ptr, particle_count, delta);
    cudaCheckError();
}

void CUDASPHBase::reset() {
    FluidSimulator::reset();

    thrust::fill(density.begin(), density.end(), 0);
    thrust::fill(velocity.begin(), velocity.end(), glm::vec3{0});
    thrust::fill(non_pressure_accel.begin(), non_pressure_accel.end(), glm::vec3{0});
}
