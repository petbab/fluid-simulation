#pragma once

#include "../math.cuh"


__device__ inline float r_to_q(float4 r, float support_radius) {
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

// Returns coefficient that should be multiplied with the original vector to obtain the gradient.
__device__ __host__ inline float cubic_spline_grad(float q, float support_radius) {
    if (q < 1.e-9 || q > 1.)
        return 0.f;

    const float grad_factor = 48.f / (PI * support_radius * support_radius * support_radius);
    const float r_len = q * support_radius;

    if (q <= 0.5)
        return grad_factor * q * (3.f*q - 2.f) / (support_radius * r_len);
    return -grad_factor * (1.f - q) * (1.f - q) / (support_radius * r_len);
}

__device__ __host__ inline float spiky(float q, float support_radius) {
    if (q > 1.f)
        return 0.f;

    float factor = 15.f / (PI * powf(support_radius, 9.f));
    return factor * powf(1 - q, 3.f);
}

// Returns coefficient that should be multiplied with the original vector to obtain the gradient.
__device__ __host__ inline float spiky_grad(float q, float support_radius) {
    if (q < 1.e-9 || q > 1.)
        return 0.f;

    float factor = -45.f / (PI * powf(support_radius, 9.f));

    float r_len = q * support_radius;
    return factor * (1.f - q) * (1.f - q) / (support_radius * r_len);
}

__device__ __host__ inline float cohesion(float q, float support_radius) {
    const float factor = 32.f / (PI * support_radius*support_radius*support_radius);

    const float iq3 = powf(1 - q, 3.f);
    const float q3 = q*q*q;
    if (q <= 0.5f)
        return factor * (2.f * q3 * iq3 - 1.f / 64.f);
    if (q <= 1.f)
        return factor * q3 * iq3;
    return 0.f;
}
