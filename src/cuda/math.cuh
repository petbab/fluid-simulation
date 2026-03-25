#pragma once


static constexpr float PI = 3.1415926535897932384626433f;

inline __device__ __host__ float4 make_float4(float x) {
    return float4{x, x, x, 0.};
}

inline __device__ __host__ float4& operator+=(float4 &a, float4 b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    return a;
}

inline __device__ __host__ float4 operator+(float4 a, float4 b) {
    return a += b;
}

inline __device__ __host__ float4& operator-=(float4 &a, float4 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
    return a;
}

inline __device__ __host__ float4 operator-(float4 a, float4 b) {
    return a -= b;
}

inline __device__ __host__ float4& operator*=(float4 &a, float b) {
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
    return a;
}

inline __device__ __host__ float4 operator*(float4 a, float b) {
    return a *= b;
}

inline __device__ __host__ float4 operator*(float b, float4 a) {
    return a *= b;
}

inline __device__ __host__ float4& operator/=(float4 &a, float b) {
    a.x /= b; a.y /= b; a.z /= b; a.w /= b;
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

struct mat4 {
#ifdef NOT_IN_KTT
    mat4() = default;
    mat4(const glm::mat4 &m) {
        col[0] = make_float4(m[0][0], m[0][1], m[0][2], m[0][3]);
        col[1] = make_float4(m[1][0], m[1][1], m[1][2], m[1][3]);
        col[2] = make_float4(m[2][0], m[2][1], m[2][2], m[2][3]);
        col[3] = make_float4(m[3][0], m[3][1], m[3][2], m[3][3]);
    }
#endif

    float4 col[4]{};
};

inline __device__ __host__ float4 operator*(mat4 m, float4 v) {
    return make_float4(
        m.col[0].x * v.x + m.col[1].x * v.y + m.col[2].x * v.z + m.col[3].x * v.w,
        m.col[0].y * v.x + m.col[1].y * v.y + m.col[2].y * v.z + m.col[3].y * v.w,
        m.col[0].z * v.x + m.col[1].z * v.y + m.col[2].z * v.z + m.col[3].z * v.w,
        m.col[0].w * v.x + m.col[1].w * v.y + m.col[2].w * v.z + m.col[3].w * v.w
    );
}
