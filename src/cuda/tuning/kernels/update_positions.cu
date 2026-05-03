#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


static constexpr float OFFSET = 0.000f;
static constexpr float COLLISION_BUFFER_MULT = 0.75f;
static constexpr float ELASTICITY = 0.5f;

__device__ void resolve_collisions(float4* positions, float4* velocities, const BoundingBoxGPU &bb) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

    float4 pos = bb.model_inv * positions[i];
    float4 vel = bb.model_inv * velocities[i];
    bool changed_pos = false;
    if (pos.x - PARTICLE_RADIUS * COLLISION_BUFFER_MULT < bb.min.x) {
        pos.x = bb.min.x + PARTICLE_RADIUS + OFFSET;
        vel.x *= -ELASTICITY;
        changed_pos = true;
    } else if (pos.x + PARTICLE_RADIUS * COLLISION_BUFFER_MULT > bb.max.x) {
        pos.x = bb.max.x - PARTICLE_RADIUS - OFFSET;
        vel.x *= -ELASTICITY;
        changed_pos = true;
    }
    if (pos.y - PARTICLE_RADIUS * COLLISION_BUFFER_MULT < bb.min.y) {
        pos.y = bb.min.y + PARTICLE_RADIUS + OFFSET;
        vel.y *= -ELASTICITY;
        changed_pos = true;
    } else if (pos.y + PARTICLE_RADIUS * COLLISION_BUFFER_MULT > bb.max.y) {
        pos.y = bb.max.y - PARTICLE_RADIUS - OFFSET;
        vel.y *= -ELASTICITY;
        changed_pos = true;
    }
    if (pos.z - PARTICLE_RADIUS * COLLISION_BUFFER_MULT < bb.min.z) {
        pos.z = bb.min.z + PARTICLE_RADIUS + OFFSET;
        vel.z *= -ELASTICITY;
        changed_pos = true;
    } else if (pos.z + PARTICLE_RADIUS * COLLISION_BUFFER_MULT > bb.max.z) {
        pos.z = bb.max.z - PARTICLE_RADIUS - OFFSET;
        vel.z *= -ELASTICITY;
        changed_pos = true;
    }

    if (changed_pos) {
        positions[i] = bb.model * pos;
        velocities[i] = bb.model * vel;
    }
}

__global__ void update_positions(float4* positions, float4* velocities, const float4* acceleration, unsigned n, float delta, float np_delta, BoundingBoxGPU bb) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    velocities[i] += np_delta * acceleration[i];
    positions[i] += delta * velocities[i];

    resolve_collisions(positions, velocities, bb);
}
