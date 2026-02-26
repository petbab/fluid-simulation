#include "sph_base.cuh"
#include "kernel.cuh"


namespace kernels {

__global__ void compute_densities_k(
    const float* positions, float* densities,
    const float *boundary_mass,
    unsigned n, const NSearch *dev_n_search
) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float4 xi = get_pos(positions, i);

    float density = cubic_spline(0.f, CUDASPHBase<>::SUPPORT_RADIUS);
    float boundary_density = 0.f;
    dev_n_search->for_neighbors(xi, [=, &density, &boundary_density] __device__ (unsigned j) {
        float4 xj = get_pos(positions, j);

        if (is_neighbor(xi, xj, i, j)) {
            float q = r_to_q(xi - xj, CUDASPHBase<>::SUPPORT_RADIUS);
            float W = cubic_spline(q, CUDASPHBase<>::SUPPORT_RADIUS);

            if (is_boundary(j, n))
                boundary_density += W * get_mass(boundary_mass, j, n);
            else
                density += W;
        }
    });
    densities[i] = density * CUDASPHBase<>::PARTICLE_MASS + boundary_density;
}

static constexpr float OFFSET = 0.000f;
static constexpr float COLLISION_BUFFER_MULT = 0.75f;

__device__ void resolve_collisions(float* positions, float4* velocities, BoundingBox bb) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    float4 pos_f4 = get_pos(positions, i);
    glm::vec4 pos = bb.model_inv * glm::vec4{pos_f4.x, pos_f4.y, pos_f4.z, 1.};
    float4 vel_f4 = velocities[i];
    glm::vec4 vel = bb.model_inv * glm::vec4{vel_f4.x, vel_f4.y, vel_f4.z, 0.};
    bool changed_pos = false;
    if (pos.x - CUDASPHBase<>::PARTICLE_RADIUS * COLLISION_BUFFER_MULT < bb.min.x) {
        pos.x = bb.min.x + CUDASPHBase<>::PARTICLE_RADIUS + OFFSET;
        vel.x *= -CUDASPHBase<>::ELASTICITY;
        changed_pos = true;
    } else if (pos.x + CUDASPHBase<>::PARTICLE_RADIUS * COLLISION_BUFFER_MULT > bb.max.x) {
        pos.x = bb.max.x - CUDASPHBase<>::PARTICLE_RADIUS - OFFSET;
        vel.x *= -CUDASPHBase<>::ELASTICITY;
        changed_pos = true;
    }
    if (pos.y - CUDASPHBase<>::PARTICLE_RADIUS * COLLISION_BUFFER_MULT < bb.min.y) {
        pos.y = bb.min.y + CUDASPHBase<>::PARTICLE_RADIUS + OFFSET;
        vel.y *= -CUDASPHBase<>::ELASTICITY;
        changed_pos = true;
    } else if (pos.y + CUDASPHBase<>::PARTICLE_RADIUS * COLLISION_BUFFER_MULT > bb.max.y) {
        pos.y = bb.max.y - CUDASPHBase<>::PARTICLE_RADIUS - OFFSET;
        vel.y *= -CUDASPHBase<>::ELASTICITY;
        changed_pos = true;
    }
    if (pos.z - CUDASPHBase<>::PARTICLE_RADIUS * COLLISION_BUFFER_MULT < bb.min.z) {
        pos.z = bb.min.z + CUDASPHBase<>::PARTICLE_RADIUS + OFFSET;
        vel.z *= -CUDASPHBase<>::ELASTICITY;
        changed_pos = true;
    } else if (pos.z + CUDASPHBase<>::PARTICLE_RADIUS * COLLISION_BUFFER_MULT > bb.max.z) {
        pos.z = bb.max.z - CUDASPHBase<>::PARTICLE_RADIUS - OFFSET;
        vel.z *= -CUDASPHBase<>::ELASTICITY;
        changed_pos = true;
    }

    if (changed_pos) {
        pos = bb.model * pos;
        set_pos(positions, i, make_float4(pos.x, pos.y, pos.z, 1.));
        vel = bb.model * vel;
        velocities[i] = make_float4(vel.x, vel.y, vel.z, 1.);
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
            float q = r_to_q(x_ij, CUDASPHBase<>::SUPPORT_RADIUS);

            velocity_laplacian += dot(v_ij, x_ij) * cubic_spline_grad(q, CUDASPHBase<>::SUPPORT_RADIUS)
                * x_ij / (densities[j] * (dot(x_ij, x_ij)
                    + 0.01f * CUDASPHBase<>::SUPPORT_RADIUS * CUDASPHBase<>::SUPPORT_RADIUS));
        }
    });

    velocity_laplacian *= 10 * CUDASPHBase<>::PARTICLE_MASS;
    acceleration[i] += CUDASPHBase<>::VISCOSITY * velocity_laplacian;
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
            float q = r_to_q(x_ij, CUDASPHBase<>::SUPPORT_RADIUS);
            if (dot(x_ij, x_ij) > 1e-6)
                f += (CUDASPHBase<>::PARTICLE_MASS * normalize(x_ij)
                    * cohesion(q, CUDASPHBase<>::SUPPORT_RADIUS) + ni - normals[j]) / (di + densities[j]);
        }
    });

    acceleration[i] -= CUDASPHBase<>::SURFACE_TENSION_ALPHA * 2.f * CUDASPHBase<>::REST_DENSITY * f;
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
            float q = r_to_q(r, CUDASPHBase<>::SUPPORT_RADIUS);
            normal += r * cubic_spline_grad(q, CUDASPHBase<>::SUPPORT_RADIUS) / densities[j];
        }
    });

    normals[i] = CUDASPHBase<>::SUPPORT_RADIUS * CUDASPHBase<>::PARTICLE_MASS * normal;
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
    float sum = cubic_spline(0.f, CUDASPHBase<>::SUPPORT_RADIUS);
    dev_n_search->for_neighbors(xi, [=, &sum] __device__ (unsigned j) {
        if (!is_boundary(j, fluid_n))
            return;

        float4 xj = get_pos(positions, j);
        if (is_neighbor(xi, xj, i, j)) {
            float4 r = xi - xj;
            float q = r_to_q(r, CUDASPHBase<>::SUPPORT_RADIUS);
            sum += cubic_spline(q, CUDASPHBase<>::SUPPORT_RADIUS);
        }
    });

    masses[i] = CUDASPHBase<>::REST_DENSITY / sum;
}

}
