#pragma once

#ifdef NOT_IN_KTT
#include <glm/glm.hpp>
#endif


/** @brief Mathematical constant pi. */
static constexpr float PI = 3.1415926535897932384626433f;

/**
 * @brief Constructs a float4 with all components set to x.
 * @param x Value to broadcast.
 * @return float4{x, x, x, 0.}.
 */
inline __device__ __host__ float4 make_float4(float x) {
    return float4{x, x, x, 0.};
}

/** @brief In-place addition for float4. */
inline __device__ __host__ float4& operator+=(float4 &a, float4 b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    return a;
}

/** @brief Component-wise addition for float4. */
inline __device__ __host__ float4 operator+(float4 a, float4 b) {
    return a += b;
}

/** @brief In-place subtraction for float4. */
inline __device__ __host__ float4& operator-=(float4 &a, float4 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
    return a;
}

/** @brief Component-wise subtraction for float4. */
inline __device__ __host__ float4 operator-(float4 a, float4 b) {
    return a -= b;
}

/** @brief In-place scalar multiplication for float4. */
inline __device__ __host__ float4& operator*=(float4 &a, float b) {
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
    return a;
}

/** @brief Scalar multiplication for float4. */
inline __device__ __host__ float4 operator*(float4 a, float b) {
    return a *= b;
}

/** @brief Scalar multiplication for float4 (commutative). */
inline __device__ __host__ float4 operator*(float b, float4 a) {
    return a *= b;
}

/** @brief In-place scalar division for float4. */
inline __device__ __host__ float4& operator/=(float4 &a, float b) {
    a.x /= b; a.y /= b; a.z /= b; a.w /= b;
    return a;
}

/** @brief Scalar division for float4. */
inline __device__ __host__ float4 operator/(float4 a, float b) {
    return a /= b;
}

/**
 * @brief Dot product of two float4 vectors (ignores w).
 * @param a First vector.
 * @param b Second vector.
 * @return Scalar dot product.
 */
inline __device__ __host__ float dot(float4 a, float4 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

/**
 * @brief Length (magnitude) of a float4 vector.
 * @param v Input vector.
 * @return Euclidean length.
 */
inline __device__ __host__ float length(float4 v) {
    return sqrtf(dot(v, v));
}

/**
 * @brief Normalizes a float4 vector.
 * @param v Input vector.
 * @return Unit vector in the same direction.
 */
inline __device__ __host__ float4 normalize(float4 v) {
    return v / length(v);
}

/**
 * @brief Simple 4x4 matrix stored by columns.
 *
 * Compatible with float4 vectors and convertible from glm::mat4 when
 * NOT_IN_KTT is defined.
 */
struct mat4 {
#ifdef NOT_IN_KTT
    mat4() = default;

    /**
     * @brief Converts from glm::mat4.
     * @param m Source glm matrix.
     */
    mat4(const glm::mat4 &m) {
        col[0] = make_float4(m[0][0], m[0][1], m[0][2], m[0][3]);
        col[1] = make_float4(m[1][0], m[1][1], m[1][2], m[1][3]);
        col[2] = make_float4(m[2][0], m[2][1], m[2][2], m[2][3]);
        col[3] = make_float4(m[3][0], m[3][1], m[3][2], m[3][3]);
    }
#endif

    float4 col[4]{};  ///< Column vectors.
};

/**
 * @brief Matrix-vector multiplication.
 * @param m 4x4 matrix.
 * @param v 4-component vector.
 * @return Transformed vector.
 */
inline __device__ __host__ float4 operator*(mat4 m, float4 v) {
    return make_float4(
        m.col[0].x * v.x + m.col[1].x * v.y + m.col[2].x * v.z + m.col[3].x * v.w,
        m.col[0].y * v.x + m.col[1].y * v.y + m.col[2].y * v.z + m.col[3].y * v.w,
        m.col[0].z * v.x + m.col[1].z * v.y + m.col[2].z * v.z + m.col[3].z * v.w,
        m.col[0].w * v.x + m.col[1].w * v.y + m.col[2].w * v.z + m.col[3].w * v.w
    );
}
