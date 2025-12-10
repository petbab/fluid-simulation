#define PI 3.1415926535f
#define PARTICLE_SPACING 0.04f
#define SUPPORT_RADIUS (2.f * PARTICLE_SPACING)
#define REST_DENSITY 1000.f
#define PARTICLE_VOLUME (PARTICLE_SPACING * PARTICLE_SPACING * PARTICLE_SPACING * 0.8f)
#define PARTICLE_MASS (REST_DENSITY * PARTICLE_VOLUME)


struct vec3 {
    float x, y, z;
};

__device__ inline vec3 operator-(vec3 a, vec3 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

__device__ inline float dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float length(vec3 a) {
    return sqrtf(dot(a, a));
}

__device__ inline vec3 get_pos(const float *positions, unsigned i) {
    unsigned ii = 3 * i;
    return {positions[ii], positions[ii + 1], positions[ii + 2]};
}

__device__ inline void set_pos(float *positions, unsigned i, vec3 pos) {
    unsigned ii = 3 * i;
    positions[ii] = pos.x;
    positions[ii + 1] = pos.y;
    positions[ii + 2] = pos.z;
}

__device__ inline bool is_neighbor(vec3 xi, vec3 xj, unsigned i, unsigned j) {
    vec3 r = xi - xj;
    return i != j && dot(r, r) <= SUPPORT_RADIUS * SUPPORT_RADIUS;
}

__device__ inline float r_to_q(vec3 r, float support_radius) {
    return length(r) / support_radius;
}

__device__ inline float cubic_spline(float q, float support_radius) {
    const float factor = 8.f / (PI * support_radius * support_radius * support_radius);
    if (q <= 0.5f)
        return factor * (6.f*q*q*q - 6.f*q*q + 1);
    if (q <= 1.f)
        return factor * 2.f * powf(1.f - q, 3.f);
    return 0.f;
}
