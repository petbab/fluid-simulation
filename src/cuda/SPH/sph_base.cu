#include "sph_base.cuh"
#include "../../debug.h"
#include "kernel.cuh"


///////////////////////////////////////////////////////////////////////////////
////                        KERNEL HYPERPARAMETERS                         ////
///////////////////////////////////////////////////////////////////////////////

static constexpr dim3 COMPUTE_DENSITY_BLOCK_SIZE = {128};
static constexpr dim3 UPDATE_POSITIONS_BLOCK_SIZE = {128};
static constexpr dim3 UPDATE_VELOCITIES_BLOCK_SIZE = {128};
static constexpr dim3 COMPUTE_XSPH_BLOCK_SIZE = {128};
static constexpr dim3 COMPUTE_VISCOSITY_BLOCK_SIZE = {128};
static constexpr dim3 COMPUTE_SURFACE_TENSION_BLOCK_SIZE = {128};
static constexpr dim3 COMPUTE_SURFACE_NORMALS_BLOCK_SIZE = {128};

///////////////////////////////////////////////////////////////////////////////
////                                KERNELS                                ////
///////////////////////////////////////////////////////////////////////////////

__global__ void compute_densities_k(
    const float* positions, float* densities,
    unsigned n, const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    glm::vec3 xi = get_pos(positions, i);

    float density = cubic_spline(0.f, CUDASPHBase::SUPPORT_RADIUS);
    dev_n_search->for_neighbors(xi, [=, &density] __device__ (unsigned j) {
        glm::vec3 xj = get_pos(positions, j);

        if (is_neighbor(xi, xj, i, j)) {
            float q = r_to_q(xi - xj, CUDASPHBase::SUPPORT_RADIUS);
            density += cubic_spline(q, CUDASPHBase::SUPPORT_RADIUS);
        }
    });
    densities[i] = density * CUDASPHBase::PARTICLE_MASS;
}

__device__ void resolve_collisions(float* positions, glm::vec3* velocities, BoundingBox bb) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

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

__global__ void update_positions_k(float* positions, glm::vec3* velocities, unsigned n, float delta, BoundingBox bb) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    glm::vec3 pos = get_pos(positions, i);
    pos += delta * velocities[i];
    set_pos(positions, i, pos);

    resolve_collisions(positions, velocities, bb);
}

__global__ void update_velocities_k(glm::vec3* velocities, const glm::vec3* acceleration, unsigned n, float delta) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    velocities[i] += acceleration[i] * delta;
}

__global__ void compute_viscosity_k(
    const float* positions, const glm::vec3* velocities,
    const float* densities, glm::vec3* acceleration,
    unsigned n, const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    glm::vec3 velocity_laplacian{0.f};
    glm::vec3 xi = get_pos(positions, i);
    glm::vec3 vi = velocities[i];

    dev_n_search->for_neighbors(xi, [=, &velocity_laplacian] __device__ (unsigned j) {
        glm::vec3 xj = get_pos(positions, j);

        if (is_neighbor(xi, xj, i, j)) {
            glm::vec3 x_ij = xi - xj;
            glm::vec3 v_ij = vi - velocities[j];
            float q = r_to_q(x_ij, CUDASPHBase::SUPPORT_RADIUS);

            velocity_laplacian += glm::dot(v_ij, x_ij) * cubic_spline_grad(q, CUDASPHBase::SUPPORT_RADIUS)
                * x_ij / (densities[j] * (glm::dot(x_ij, x_ij)
                    + 0.01f * CUDASPHBase::SUPPORT_RADIUS * CUDASPHBase::SUPPORT_RADIUS));
        }
    });

    velocity_laplacian *= 10 * CUDASPHBase::PARTICLE_MASS;
    acceleration[i] += CUDASPHBase::VISCOSITY * velocity_laplacian;
}

__global__ void compute_surface_tension_k(
    const float* positions, const float* densities,
    const glm::vec3* normals, glm::vec3* acceleration,
    unsigned n, const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    glm::vec3 xi = get_pos(positions, i);
    glm::vec3 ni = normals[i];
    float di = densities[i];
    glm::vec3 f{0.f};

    dev_n_search->for_neighbors(xi, [=, &f] __device__ (unsigned j) {
        glm::vec3 xj = get_pos(positions, j);

        if (is_neighbor(xi, xj, i, j)) {
            glm::vec3 x_ij = xi - get_pos(positions, j);
            float q = r_to_q(x_ij, CUDASPHBase::SUPPORT_RADIUS);
            if (glm::dot(x_ij, x_ij) > 1e-6)
                f += (CUDASPHBase::PARTICLE_MASS * glm::normalize(x_ij)
                    * cohesion(q, CUDASPHBase::SUPPORT_RADIUS) + ni - normals[j]) / (di + densities[j]);
        }
    });

    acceleration[i] -= CUDASPHBase::SURFACE_TENSION_ALPHA * 2.f * CUDASPHBase::REST_DENSITY * f;
}

__global__ void compute_surface_normals_k(
    const float* positions, const float* densities,
    glm::vec3* normals, unsigned n,
    const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    glm::vec3 xi = get_pos(positions, i);
    glm::vec3 normal{0.f};

    dev_n_search->for_neighbors(xi, [=, &normal] __device__ (unsigned j) {
        glm::vec3 xj = get_pos(positions, j);

        if (is_neighbor(xi, xj, i, j)) {
            glm::vec3 r = xi - xj;
            float q = r_to_q(r, CUDASPHBase::SUPPORT_RADIUS);
            normal += r * cubic_spline_grad(q, CUDASPHBase::SUPPORT_RADIUS) / densities[j];
        }
    });

    normals[i] = CUDASPHBase::SUPPORT_RADIUS * CUDASPHBase::PARTICLE_MASS * normal;
}

///////////////////////////////////////////////////////////////////////////////
////                              CUDASPHBase                              ////
///////////////////////////////////////////////////////////////////////////////

CUDASPHBase::CUDASPHBase(unsigned grid_count, const BoundingBox& bounding_box, bool is_2d)
    : CUDASimulator(grid_count, bounding_box, is_2d),
      density(particle_count),
      velocity(particle_count),
      n_search{2.f * SUPPORT_RADIUS},
      non_pressure_accel(particle_count),
      normal(particle_count) {
}

void CUDASPHBase::compute_densities(const float* positions_dev_ptr) {
    float* densities_ptr = thrust::raw_pointer_cast(density.data());

    const dim3 grid_size{particle_count / COMPUTE_DENSITY_BLOCK_SIZE.x + 1};
    compute_densities_k<<<COMPUTE_DENSITY_BLOCK_SIZE, grid_size>>>(
        positions_dev_ptr, densities_ptr, particle_count, n_search.dev_ptr());
    cudaCheckError();
}

void CUDASPHBase::update_positions(float* positions_dev_ptr, float delta) {
    glm::vec3* velocities_ptr = thrust::raw_pointer_cast(velocity.data());

    const dim3 grid_size{particle_count / UPDATE_POSITIONS_BLOCK_SIZE.x + 1};
    update_positions_k<<<UPDATE_POSITIONS_BLOCK_SIZE, grid_size>>>(
        positions_dev_ptr, velocities_ptr, particle_count, delta, bounding_box);
    cudaCheckError();
}

void CUDASPHBase::apply_non_pressure_forces(const float* positions_dev_ptr, float delta) {
    thrust::fill(non_pressure_accel.begin(), non_pressure_accel.end(), GRAVITY);

    compute_viscosity(positions_dev_ptr);
    compute_surface_tension(positions_dev_ptr);

    update_velocities(delta);

    compute_XSPH(positions_dev_ptr);
}

void CUDASPHBase::reset() {
    FluidSimulator::reset();

    thrust::fill(density.begin(), density.end(), 0);
    thrust::fill(velocity.begin(), velocity.end(), glm::vec3{0});
    thrust::fill(non_pressure_accel.begin(), non_pressure_accel.end(), glm::vec3{0});
    thrust::fill(normal.begin(), normal.end(), glm::vec3{0});
}

void CUDASPHBase::update_velocities(float delta) {
    delta = std::min(delta, NON_PRESSURE_MAX_TIME_STEP);

    glm::vec3* velocities_ptr = thrust::raw_pointer_cast(velocity.data());
    const glm::vec3* acceleration_ptr = thrust::raw_pointer_cast(non_pressure_accel.data());

    const dim3 grid_size{particle_count / UPDATE_VELOCITIES_BLOCK_SIZE.x + 1};
    update_velocities_k<<<UPDATE_VELOCITIES_BLOCK_SIZE, grid_size>>>(
        velocities_ptr, acceleration_ptr, particle_count, delta);
    cudaCheckError();
}

void CUDASPHBase::compute_XSPH(const float* positions_dev_ptr) {
    if constexpr (XSPH_ALPHA == 0.f)
        return;
    // TODO
}

void CUDASPHBase::compute_viscosity(const float* positions_dev_ptr) {
    if constexpr (VISCOSITY == 0.f)
        return;

    const float* densities_ptr = thrust::raw_pointer_cast(density.data());
    const glm::vec3* velocities_ptr = thrust::raw_pointer_cast(velocity.data());
    glm::vec3* acceleration_ptr = thrust::raw_pointer_cast(non_pressure_accel.data());

    const dim3 grid_size{particle_count / COMPUTE_VISCOSITY_BLOCK_SIZE.x + 1};
    compute_viscosity_k<<<COMPUTE_VISCOSITY_BLOCK_SIZE, grid_size>>>(
        positions_dev_ptr, velocities_ptr, densities_ptr,
        acceleration_ptr, particle_count, n_search.dev_ptr());
    cudaCheckError();
}

void CUDASPHBase::compute_surface_tension(const float* positions_dev_ptr) {
    compute_surface_normals(positions_dev_ptr);

    const float* densities_ptr = thrust::raw_pointer_cast(density.data());
    const glm::vec3* normals_ptr = thrust::raw_pointer_cast(normal.data());
    glm::vec3* acceleration_ptr = thrust::raw_pointer_cast(non_pressure_accel.data());

    const dim3 grid_size{particle_count / COMPUTE_SURFACE_TENSION_BLOCK_SIZE.x + 1};
    compute_surface_tension_k<<<COMPUTE_SURFACE_TENSION_BLOCK_SIZE, grid_size>>>(
        positions_dev_ptr, densities_ptr, normals_ptr,
        acceleration_ptr, particle_count, n_search.dev_ptr());
    cudaCheckError();
}

void CUDASPHBase::compute_surface_normals(const float* positions_dev_ptr) {
    const float* densities_ptr = thrust::raw_pointer_cast(density.data());
    glm::vec3* normals_ptr = thrust::raw_pointer_cast(normal.data());

    const dim3 grid_size{particle_count / COMPUTE_SURFACE_NORMALS_BLOCK_SIZE.x + 1};
    compute_surface_normals_k<<<COMPUTE_SURFACE_NORMALS_BLOCK_SIZE, grid_size>>>(
        positions_dev_ptr, densities_ptr, normals_ptr,
        particle_count, n_search.dev_ptr());
    cudaCheckError();
}
