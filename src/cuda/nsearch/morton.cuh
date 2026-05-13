#pragma once


/** @brief Morton code type (64-bit unsigned integer). */
using morton_t = unsigned long long;

/**
 * @brief Spreads bits of n by inserting two zeros between each bit.
 *
 * Part of the 3D Morton code encoding pipeline.
 * @param n 21-bit input value.
 * @return Bit-spread value.
 */
__device__ __host__ inline morton_t part_by_3(morton_t n) {
    n = n & 0x1fffff;  // Limit to 21 bits (for 64-bit Morton codes)
    n = (n | (n << 32)) & 0x1f00000000ffff;
    n = (n | (n << 16)) & 0x1f0000ff0000ff;
    n = (n | (n << 8))  & 0x100f00f00f00f00f;
    n = (n | (n << 4))  & 0x10c30c30c30c30c3;
    n = (n | (n << 2))  & 0x1249249249249249;
    return n;
}

/**
 * @brief Converts a signed integer to unsigned by offsetting.
 *
 * Offset allows encoding of negative coordinates into unsigned Morton codes.
 * @param n Signed integer.
 * @return Unsigned offset integer.
 */
__device__ __host__ inline unsigned signed_to_unsigned(int n) {
    unsigned offset = 1u << 20u;
    return static_cast<unsigned>(n) + offset;
}

/**
 * @brief Converts an unsigned integer back to signed.
 * @param n Unsigned integer.
 * @return Signed integer.
 */
__device__ __host__ inline int unsigned_to_signed(unsigned n) {
    int offset = 1u << 20u;
    return static_cast<int>(n) - offset;
}

/**
 * @brief Encodes a 3D cell coordinate into a Morton code.
 *
 * Coordinate range: -2^20 <= c <= 2^20 - 1.
 * @param c Cell coordinates (x, y, z).
 * @return 64-bit Morton code.
 */
__device__ __host__ inline morton_t encode_morton(int3 c) {
    return part_by_3(signed_to_unsigned(c.x))
        | (part_by_3(signed_to_unsigned(c.y)) << 1)
        | (part_by_3(signed_to_unsigned(c.z)) << 2);
}
