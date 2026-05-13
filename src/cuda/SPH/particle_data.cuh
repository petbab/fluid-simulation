#pragma once

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <glad/glad.h>
#include <cuda/cuda_gl_buffer.h>
#include <cuda/nsearch/morton.cuh>
#include <debug.h>


/**
 * @brief Double-buffered device vector for ping-pong buffering.
 * @tparam T Element type.
 */
template<typename T>
class DoubleBuffer {
public:
    /**
     * @brief Constructs a double buffer with n elements.
     * @param n Number of elements per buffer.
     */
    explicit DoubleBuffer(unsigned n)
        : a{n}, b{n}, src_p{&a}, dst_p {&b} {}

    /** @return Reference to the current source buffer. */
    thrust::device_vector<T>& src() { return *src_p; }
    /** @return Reference to the current destination buffer. */
    thrust::device_vector<T>& dst() { return *dst_p; }

    /** @return Const reference to the current source buffer. */
    const thrust::device_vector<T>& src() const { return *src_p; }
    /** @return Const reference to the current destination buffer. */
    const thrust::device_vector<T>& dst() const { return *dst_p; }

    /** @brief Swaps source and destination pointers. */
    void swap() {
        std::swap(src_p, dst_p);
    }

private:
    thrust::device_vector<T> a, b;
    thrust::device_vector<T> *src_p, *dst_p;
};

/**
 * @brief Double-buffered CUDA-GL buffer for ping-pong position rendering.
 */
class DoubleGLBuffer {
public:
    /**
     * @brief Constructs a double GL buffer.
     * @param vao_a OpenGL VAO/buffer A.
     * @param vao_b OpenGL VAO/buffer B.
     */
    DoubleGLBuffer(GLuint vao_a, GLuint vao_b)
        : a{vao_a}, b{vao_b}, src_p{&a}, dst_p {&b} {}

    /** @return Lock for the current source buffer. */
    CUDAGLBuffer::CUDALock lock_src() const { return src_p->lock(); }
    /** @return Lock for the current destination buffer. */
    CUDAGLBuffer::CUDALock lock_dst() const { return dst_p->lock(); }

    /** @brief Swaps source and destination pointers. */
    void swap() {
        std::swap(src_p, dst_p);
    }

private:
    CUDAGLBuffer a, b;
    CUDAGLBuffer *src_p, *dst_p;
};

/**
 * @brief Manages all per-particle simulation data on the GPU.
 *
 * Stores positions, velocities, densities, pressures, accelerations, and
 * neighbor search metadata. Supports Morton-code sorting of particles
 * for spatial locality.
 */
class ParticleData {
    template <class T>
    static T* dev_ptr(thrust::device_vector<T>& vec) {
        return thrust::raw_pointer_cast(vec.data());
    }
    template <class T>
    static const T* dev_ptr(const thrust::device_vector<T>& vec) {
        return thrust::raw_pointer_cast(vec.data());
    }

public:
    /**
     * @brief Constructs particle data buffers.
     * @param fluid_n Number of fluid particles.
     * @param boundary_n Number of boundary particles.
     * @param cell_size Spatial cell size for sorting.
     */
    ParticleData(unsigned fluid_n, unsigned boundary_n, float cell_size)
        : fluid_n{fluid_n}, boundary_n{boundary_n},
        cell_size{cell_size}, cell_size_mult{1.f},
        morton_codes(fluid_n),
        boundary_morton_codes(boundary_n),
        indices(fluid_n),
        boundary_indices(boundary_n),
        density_buf(fluid_n),
        boundary_mass_buf(boundary_n),
        pressure_buf(fluid_n),
        velocity_buf(fluid_n),
        non_pressure_accel_buf(fluid_n),
        pressure_accel_buf(fluid_n),
        normal_buf(fluid_n) {}

    /**
     * @brief Initializes the double-buffered position GL buffers.
     * @param pos_vao_a OpenGL buffer A.
     * @param pos_vao_b OpenGL buffer B.
     */
    void init_positions(GLuint pos_vao_a, GLuint pos_vao_b) {
        position_buf = std::make_unique<DoubleGLBuffer>(pos_vao_a, pos_vao_b);
    }

    /**
     * @brief Sets the cell size multiplier for sorting.
     * @param cs_mult Multiplier applied to cell_size during Morton encoding.
     */
    void set_cell_size_mult(float cs_mult) { cell_size_mult = cs_mult; }

    /**
     * @brief Sorts boundary particles by Morton code.
     * @param positions_src Source positions (device).
     * @param positions_dst Destination positions (device).
     */
    void sort_boundary(float4 *positions_src, float4 *positions_dst) {
        assert(boundary_n > 0);

        thrust::sequence(boundary_indices.begin(), boundary_indices.end());

        const thrust::device_ptr<float4> pos_src{positions_src + fluid_n};
        const float cs = cell_size * cell_size_mult;
        thrust::transform(pos_src, pos_src + boundary_n, boundary_morton_codes.begin(),
            [cs] __device__ (float4 p) -> morton_t {
                int3 cell_coord{
                    static_cast<int>(floorf(p.x / cs)),
                    static_cast<int>(floorf(p.y / cs)),
                    static_cast<int>(floorf(p.z / cs))
                };
                return encode_morton(cell_coord);
            });

        thrust::sort_by_key(boundary_morton_codes.begin(), boundary_morton_codes.end(), boundary_indices.begin());

        const thrust::device_ptr<float4> pos_dst{positions_dst + fluid_n};
        thrust::gather(boundary_indices.begin(), boundary_indices.end(), pos_src, pos_dst);
        cudaMemcpy(pos_src.get(), pos_dst.get(), boundary_n * sizeof(float4), cudaMemcpyDeviceToDevice);
        cudaCheckError();

        // Sequence again for visualization
        thrust::sequence(boundary_indices.begin(), boundary_indices.end());
    }

    /**
     * @brief Sorts fluid particles by Morton code and swaps buffers.
     * @param positions_src Source positions (device).
     * @param positions_dst Destination positions (device).
     */
    void sort(const float4 *positions_src, float4 *positions_dst) {
        thrust::sequence(indices.begin(), indices.end());

        const thrust::device_ptr<const float4> pos_src{positions_src};
        const float cs = cell_size * cell_size_mult;
        thrust::transform(pos_src, pos_src + fluid_n, morton_codes.begin(),
            [cs] __device__ (float4 p) -> morton_t {
                int3 cell_coord{
                    static_cast<int>(floorf(p.x / cs)),
                    static_cast<int>(floorf(p.y / cs)),
                    static_cast<int>(floorf(p.z / cs))
                };
                return encode_morton(cell_coord);
            });

        thrust::sort_by_key(morton_codes.begin(), morton_codes.end(), indices.begin());

        const thrust::device_ptr<float4> pos_dst{positions_dst};
        thrust::gather(indices.begin(), indices.end(), pos_src, pos_dst);
        thrust::gather(indices.begin(), indices.end(), velocity_buf.src().begin(), velocity_buf.dst().begin());
        swap_buffers();
    }

    /** @return Const reference to the double-buffered position buffer. */
    const DoubleGLBuffer& position() const {
        assert(position_buf != nullptr);
        return *position_buf;
    }

    /** @return Device pointer to density array. */
    float* density() { return dev_ptr(density_buf); }
    /** @return Device pointer to boundary mass array. */
    float* boundary_mass() { return dev_ptr(boundary_mass_buf); }
    /** @return Device pointer to pressure array. */
    float* pressure() { return dev_ptr(pressure_buf); }
    /** @return Device pointer to velocity array (source buffer). */
    float4* velocity() { return dev_ptr(velocity_buf.src()); }
    /** @return Device pointer to non-pressure acceleration array. */
    float4* non_pressure_accel() { return dev_ptr(non_pressure_accel_buf); }
    /** @return Device pointer to pressure acceleration array. */
    float4* pressure_accel() { return dev_ptr(pressure_accel_buf); }
    /** @return Device pointer to normal array. */
    float4* normal() { return dev_ptr(normal_buf); }

    /** @return Device pointer to velocity destination buffer. */
    float4* velocity_dst() { return dev_ptr(velocity_buf.dst()); }

    /** @return Reference to density device vector. */
    thrust::device_vector<float>& density_vec() { return density_buf; }
    /** @return Reference to boundary mass device vector. */
    thrust::device_vector<float>& boundary_mass_vec() { return boundary_mass_buf; }
    /** @return Reference to pressure device vector. */
    thrust::device_vector<float>& pressure_vec() { return pressure_buf; }
    /** @return Reference to velocity device vector. */
    thrust::device_vector<float4>& velocity_vec() { return velocity_buf.src(); }
    /** @return Reference to non-pressure acceleration device vector. */
    thrust::device_vector<float4>& non_pressure_accel_vec() { return non_pressure_accel_buf; }
    /** @return Reference to pressure acceleration device vector. */
    thrust::device_vector<float4>& pressure_accel_vec() { return pressure_accel_buf; }
    /** @return Reference to normal device vector. */
    thrust::device_vector<float4>& normal_vec() { return normal_buf; }

    /** @return Const reference to density device vector. */
    const thrust::device_vector<float>& density_vec() const { return density_buf; }
    /** @return Const reference to boundary mass device vector. */
    const thrust::device_vector<float>& boundary_mass_vec() const { return boundary_mass_buf; }
    /** @return Const reference to pressure device vector. */
    const thrust::device_vector<float>& pressure_vec() const { return pressure_buf; }
    /** @return Const reference to velocity device vector. */
    const thrust::device_vector<float4>& velocity_vec() const { return velocity_buf.src(); }
    /** @return Const reference to non-pressure acceleration device vector. */
    const thrust::device_vector<float4>& non_pressure_accel_vec() const { return non_pressure_accel_buf; }
    /** @return Const reference to pressure acceleration device vector. */
    const thrust::device_vector<float4>& pressure_accel_vec() const { return pressure_accel_buf; }
    /** @return Const reference to normal device vector. */
    const thrust::device_vector<float4>& normal_vec() const { return normal_buf; }

    /** @return Device pointer to fluid particle indices. */
    unsigned* get_indices() { return dev_ptr(indices); }
    /** @return Device pointer to boundary particle indices. */
    unsigned* get_boundary_indices() { return dev_ptr(boundary_indices); }

    /** @return Const device pointer to fluid particle indices. */
    const unsigned* get_indices() const { return dev_ptr(indices); }
    /** @return Const device pointer to boundary particle indices. */
    const unsigned* get_boundary_indices() const { return dev_ptr(boundary_indices); }

private:
    void swap_buffers() {
        velocity_buf.swap();
        position_buf->swap();
    }

    unsigned fluid_n, boundary_n;  ///< Particle counts.
    float cell_size, cell_size_mult;  ///< Cell size and multiplier.
    thrust::device_vector<morton_t> morton_codes, boundary_morton_codes;  ///< Morton codes for sorting.
    thrust::device_vector<unsigned> indices, boundary_indices;  ///< Sorted indices.

    thrust::device_vector<float> density_buf, boundary_mass_buf, pressure_buf;  ///< Scalar buffers.
    thrust::device_vector<float4> non_pressure_accel_buf, pressure_accel_buf, normal_buf;  ///< Vector buffers.

    DoubleBuffer<float4> velocity_buf;  ///< Double-buffered velocities.
    std::unique_ptr<DoubleGLBuffer> position_buf;  ///< Double-buffered positions.
};
