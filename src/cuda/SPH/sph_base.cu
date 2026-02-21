#include "sph_base.cuh"
#include "../../debug.h"
#include "kernel.cuh"
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>


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
static constexpr dim3 COMPUTE_BOUNDARY_MASS_BLOCK_SIZE = {128};

///////////////////////////////////////////////////////////////////////////////
////                                KERNELS                                ////
///////////////////////////////////////////////////////////////////////////////

__global__ void compute_densities_k(
    const float* positions, float* densities,
    const float *boundary_mass,
    unsigned n, const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float4 xi = get_pos(positions, i);

    float density = cubic_spline(0.f, CUDASPHBase::SUPPORT_RADIUS);
    float boundary_density = 0.f;
    dev_n_search->for_neighbors(xi, [=, &density, &boundary_density] __device__ (unsigned j) {
        float4 xj = get_pos(positions, j);

        if (is_neighbor(xi, xj, i, j)) {
            float q = r_to_q(xi - xj, CUDASPHBase::SUPPORT_RADIUS);
            float W = cubic_spline(q, CUDASPHBase::SUPPORT_RADIUS);

            if (is_boundary(j, n))
                boundary_density += W * get_mass(boundary_mass, j, n);
            else
                density += W;
        }
    });
    densities[i] = density * CUDASPHBase::PARTICLE_MASS + boundary_density;
}

static constexpr float OFFSET = 0.000f;
static constexpr float COLLISION_BUFFER_MULT = 0.75f;

__device__ void resolve_collisions(float* positions, float4* velocities, BoundingBox bb) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    float4 pos = get_pos(positions, i);
    float4 vel = velocities[i];
    bool changed_pos = false;
    if (pos.x - CUDASPHBase::PARTICLE_RADIUS * COLLISION_BUFFER_MULT < bb.min.x) {
        pos.x = bb.min.x + CUDASPHBase::PARTICLE_RADIUS + OFFSET;
        vel.x *= -CUDASPHBase::ELASTICITY;
        changed_pos = true;
    } else if (pos.x + CUDASPHBase::PARTICLE_RADIUS * COLLISION_BUFFER_MULT > bb.max.x) {
        pos.x = bb.max.x - CUDASPHBase::PARTICLE_RADIUS - OFFSET;
        vel.x *= -CUDASPHBase::ELASTICITY;
        changed_pos = true;
    }
    if (pos.y - CUDASPHBase::PARTICLE_RADIUS * COLLISION_BUFFER_MULT < bb.min.y) {
        pos.y = bb.min.y + CUDASPHBase::PARTICLE_RADIUS + OFFSET;
        vel.y *= -CUDASPHBase::ELASTICITY;
        changed_pos = true;
    } else if (pos.y + CUDASPHBase::PARTICLE_RADIUS * COLLISION_BUFFER_MULT > bb.max.y) {
        pos.y = bb.max.y - CUDASPHBase::PARTICLE_RADIUS - OFFSET;
        vel.y *= -CUDASPHBase::ELASTICITY;
        changed_pos = true;
    }
    if (pos.z - CUDASPHBase::PARTICLE_RADIUS * COLLISION_BUFFER_MULT < bb.min.z) {
        pos.z = bb.min.z + CUDASPHBase::PARTICLE_RADIUS + OFFSET;
        vel.z *= -CUDASPHBase::ELASTICITY;
        changed_pos = true;
    } else if (pos.z + CUDASPHBase::PARTICLE_RADIUS * COLLISION_BUFFER_MULT > bb.max.z) {
        pos.z = bb.max.z - CUDASPHBase::PARTICLE_RADIUS - OFFSET;
        vel.z *= -CUDASPHBase::ELASTICITY;
        changed_pos = true;
    }

    if (changed_pos) {
        set_pos(positions, i, pos);
        velocities[i] = vel;
    }
}

__global__ void update_positions_k(float* positions, float4* velocities, unsigned n, float delta, BoundingBox bb) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float4 pos = get_pos(positions, i);
    pos += delta * velocities[i];
    set_pos(positions, i, pos);

    resolve_collisions(positions, velocities, bb);
}

__global__ void update_velocities_k(float4* velocities, const float4* acceleration, unsigned n, float delta) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    velocities[i] += acceleration[i] * delta;
}

__global__ void compute_viscosity_k(
    const float* positions, const float4* velocities,
    const float* densities, float4* acceleration,
    unsigned n, const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float4 velocity_laplacian{0.f};
    float4 xi = get_pos(positions, i);
    float4 vi = velocities[i];

    dev_n_search->for_neighbors(xi, [=, &velocity_laplacian] __device__ (unsigned j) {
        if (is_boundary(j, n))
            return;

        float4 xj = get_pos(positions, j);

        if (is_neighbor(xi, xj, i, j)) {
            float4 x_ij = xi - xj;
            float4 v_ij = vi - velocities[j];
            float q = r_to_q(x_ij, CUDASPHBase::SUPPORT_RADIUS);

            velocity_laplacian += dot(v_ij, x_ij) * cubic_spline_grad(q, CUDASPHBase::SUPPORT_RADIUS)
                * x_ij / (densities[j] * (dot(x_ij, x_ij)
                    + 0.01f * CUDASPHBase::SUPPORT_RADIUS * CUDASPHBase::SUPPORT_RADIUS));
        }
    });

    velocity_laplacian *= 10 * CUDASPHBase::PARTICLE_MASS;
    acceleration[i] += CUDASPHBase::VISCOSITY * velocity_laplacian;
}

__global__ void compute_surface_tension_k(
    const float* positions, const float* densities,
    const float4* normals, float4* acceleration,
    unsigned n, const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float4 xi = get_pos(positions, i);
    float4 ni = normals[i];
    float di = densities[i];
    float4 f{0.f};

    dev_n_search->for_neighbors(xi, [=, &f] __device__ (unsigned j) {
        if (is_boundary(j, n))
            return;

        float4 xj = get_pos(positions, j);

        if (is_neighbor(xi, xj, i, j)) {
            float4 x_ij = xi - get_pos(positions, j);
            float q = r_to_q(x_ij, CUDASPHBase::SUPPORT_RADIUS);
            if (dot(x_ij, x_ij) > 1e-6)
                f += (CUDASPHBase::PARTICLE_MASS * normalize(x_ij)
                    * cohesion(q, CUDASPHBase::SUPPORT_RADIUS) + ni - normals[j]) / (di + densities[j]);
        }
    });

    acceleration[i] -= CUDASPHBase::SURFACE_TENSION_ALPHA * 2.f * CUDASPHBase::REST_DENSITY * f;
}

__global__ void compute_surface_normals_k(
    const float* positions, const float* densities,
    float4* normals, unsigned n,
    const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float4 xi = get_pos(positions, i);
    float4 normal{0.f};

    dev_n_search->for_neighbors(xi, [=, &normal] __device__ (unsigned j) {
        if (is_boundary(j, n))
            return;

        float4 xj = get_pos(positions, j);

        if (is_neighbor(xi, xj, i, j)) {
            float4 r = xi - xj;
            float q = r_to_q(r, CUDASPHBase::SUPPORT_RADIUS);
            normal += r * cubic_spline_grad(q, CUDASPHBase::SUPPORT_RADIUS) / densities[j];
        }
    });

    normals[i] = CUDASPHBase::SUPPORT_RADIUS * CUDASPHBase::PARTICLE_MASS * normal;
}

__global__ void compute_boundary_mass_k(
    const float* positions, float* masses,
    unsigned fluid_n, unsigned boundary_n,
    const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= boundary_n)
        return;

    float4 xi = get_pos(positions, fluid_n + i);
    float sum = cubic_spline(0.f, CUDASPHBase::SUPPORT_RADIUS);
    dev_n_search->for_neighbors(xi, [=, &sum] __device__ (unsigned j) {
        if (!is_boundary(j, fluid_n))
            return;

        float4 xj = get_pos(positions, j);
        if (is_neighbor(xi, xj, i, j)) {
            float4 r = xi - xj;
            float q = r_to_q(r, CUDASPHBase::SUPPORT_RADIUS);
            sum += cubic_spline(q, CUDASPHBase::SUPPORT_RADIUS);
        }
    });

    masses[i] = CUDASPHBase::REST_DENSITY / sum;
}

///////////////////////////////////////////////////////////////////////////////
////                              CUDASPHBase                              ////
///////////////////////////////////////////////////////////////////////////////

CUDASPHBase::CUDASPHBase(const opts_t &opts)
    : CUDASimulator(opts),
      density(fluid_particles),
      boundary_mass(boundary_particles),
      velocity(fluid_particles),
      n_search{2.f * SUPPORT_RADIUS},
      non_pressure_accel(fluid_particles),
      normal(fluid_particles) {
}

void CUDASPHBase::compute_densities(const float* positions_dev_ptr) {
    float* densities_ptr = thrust::raw_pointer_cast(density.data());
    float* boundary_mass_ptr = thrust::raw_pointer_cast(boundary_mass.data());

    const dim3 grid_size{fluid_particles / COMPUTE_DENSITY_BLOCK_SIZE.x + 1};
    compute_densities_k<<<COMPUTE_DENSITY_BLOCK_SIZE, grid_size>>>(
        positions_dev_ptr, densities_ptr, boundary_mass_ptr, fluid_particles, n_search.dev_ptr());
    cudaCheckError();
}

void CUDASPHBase::update_positions(float* positions_dev_ptr, float delta) {
    float4* velocities_ptr = thrust::raw_pointer_cast(velocity.data());

    const dim3 grid_size{fluid_particles / UPDATE_POSITIONS_BLOCK_SIZE.x + 1};
    update_positions_k<<<UPDATE_POSITIONS_BLOCK_SIZE, grid_size>>>(
        positions_dev_ptr, velocities_ptr, fluid_particles, delta, bounding_box);
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
    thrust::fill(boundary_mass.begin(), boundary_mass.end(), 0);
    thrust::fill(velocity.begin(), velocity.end(), make_float4(0));
    thrust::fill(non_pressure_accel.begin(), non_pressure_accel.end(), make_float4(0));
    thrust::fill(normal.begin(), normal.end(), make_float4(0));
}

void CUDASPHBase::update_velocities(float delta) {
    delta = std::min(delta, NON_PRESSURE_MAX_TIME_STEP);

    float4* velocities_ptr = thrust::raw_pointer_cast(velocity.data());
    const float4* acceleration_ptr = thrust::raw_pointer_cast(non_pressure_accel.data());

    const dim3 grid_size{fluid_particles / UPDATE_VELOCITIES_BLOCK_SIZE.x + 1};
    update_velocities_k<<<UPDATE_VELOCITIES_BLOCK_SIZE, grid_size>>>(
        velocities_ptr, acceleration_ptr, fluid_particles, delta);
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
    const float4* velocities_ptr = thrust::raw_pointer_cast(velocity.data());
    float4* acceleration_ptr = thrust::raw_pointer_cast(non_pressure_accel.data());

    const dim3 grid_size{fluid_particles / COMPUTE_VISCOSITY_BLOCK_SIZE.x + 1};
    compute_viscosity_k<<<COMPUTE_VISCOSITY_BLOCK_SIZE, grid_size>>>(
        positions_dev_ptr, velocities_ptr, densities_ptr,
        acceleration_ptr, fluid_particles, n_search.dev_ptr());
    cudaCheckError();
}

void CUDASPHBase::compute_surface_tension(const float* positions_dev_ptr) {
    compute_surface_normals(positions_dev_ptr);

    const float* densities_ptr = thrust::raw_pointer_cast(density.data());
    const float4* normals_ptr = thrust::raw_pointer_cast(normal.data());
    float4* acceleration_ptr = thrust::raw_pointer_cast(non_pressure_accel.data());

    const dim3 grid_size{fluid_particles / COMPUTE_SURFACE_TENSION_BLOCK_SIZE.x + 1};
    compute_surface_tension_k<<<COMPUTE_SURFACE_TENSION_BLOCK_SIZE, grid_size>>>(
        positions_dev_ptr, densities_ptr, normals_ptr,
        acceleration_ptr, fluid_particles, n_search.dev_ptr());
    cudaCheckError();
}

void CUDASPHBase::compute_surface_normals(const float* positions_dev_ptr) {
    const float* densities_ptr = thrust::raw_pointer_cast(density.data());
    float4* normals_ptr = thrust::raw_pointer_cast(normal.data());

    const dim3 grid_size{fluid_particles / COMPUTE_SURFACE_NORMALS_BLOCK_SIZE.x + 1};
    compute_surface_normals_k<<<COMPUTE_SURFACE_NORMALS_BLOCK_SIZE, grid_size>>>(
        positions_dev_ptr, densities_ptr, normals_ptr,
        fluid_particles, n_search.dev_ptr());
    cudaCheckError();
}

struct float4_length_sq {
    __host__ __device__ float operator()(const float4& v) const {
        return length(v);
    }
};

float CUDASPHBase::adapt_time_step(float delta, float min_step, float max_step) const {
    float max_velocity = sqrtf(thrust::transform_reduce(
        velocity.begin(),
        velocity.end(),
        float4_length_sq(),      // unary transform
        0.0f,                    // initial value
        thrust::maximum<float>() // reduction op
    ));

    if (max_velocity < 1.e-9)
        return max_step;

    float cfl_max_time_step = CFL_FACTOR * PARTICLE_SPACING / max_velocity;

    return std::min(std::clamp(delta, min_step, max_step), cfl_max_time_step);
}

void CUDASPHBase::compute_boundary_mass(const float* positions_dev_ptr) {
    float* masses_ptr = thrust::raw_pointer_cast(boundary_mass.data());

    const dim3 grid_size{boundary_particles / COMPUTE_BOUNDARY_MASS_BLOCK_SIZE.x + 1};
    compute_boundary_mass_k<<<COMPUTE_BOUNDARY_MASS_BLOCK_SIZE, grid_size>>>(
        positions_dev_ptr, masses_ptr, fluid_particles, boundary_particles, n_search.dev_ptr());
    cudaCheckError();
}
