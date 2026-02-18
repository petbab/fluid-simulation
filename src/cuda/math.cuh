#pragma once


static constexpr float PI = 3.1415926535897932384626433f;

inline __device__ __host__ float4 make_float4(float x) {
    return float4{x, x, x, 0.};
}

inline __device__ __host__ float4& operator+=(float4 &a, float4 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

inline __device__ __host__ float4 operator+(float4 a, float4 b) {
    return a += b;
}

inline __device__ __host__ float4& operator-=(float4 &a, float4 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
    return a;
}

inline __device__ __host__ float4 operator-(float4 a, float4 b) {
    return a -= b;
}

inline __device__ __host__ float4& operator*=(float4 &a, float b) {
    a.x *= b; a.y *= b; a.z *= b;
    return a;
}

inline __device__ __host__ float4 operator*(float4 a, float b) {
    return a *= b;
}

inline __device__ __host__ float4 operator*(float b, float4 a) {
    return a *= b;
}

inline __device__ __host__ float4& operator/=(float4 &a, float b) {
    a.x /= b; a.y /= b; a.z /= b;
    return a;
}

inline __device__ __host__ float4 operator/(float4 a, float b) {
    return a /= b;
}

inline __device__ __host__ float dot(float4 a, float4 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __device__ __host__ float length(float4 v) {
    return sqrtf(dot(v, v));
}

inline __device__ __host__ float4 normalize(float4 v) {
    return v / length(v);
}
