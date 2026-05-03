#pragma once

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <glad/glad.h>
#include <cuda/cuda_gl_buffer.h>
#include <cuda/nsearch/morton.cuh>


template<typename T>
class DoubleBuffer {
public:
    explicit DoubleBuffer(unsigned n)
        : a{n}, b{n}, src_p{&a}, dst_p {&b} {}

    thrust::device_vector<T>& src() { return *src_p; }
    thrust::device_vector<T>& dst() { return *dst_p; }

    const thrust::device_vector<T>& src() const { return *src_p; }
    const thrust::device_vector<T>& dst() const { return *dst_p; }

    void swap() {
        std::swap(src_p, dst_p);
    }

private:
    thrust::device_vector<T> a, b;
    thrust::device_vector<T> *src_p, *dst_p;
};

class DoubleGLBuffer {
public:
    DoubleGLBuffer(GLuint vao_a, GLuint vao_b)
        : a{vao_a}, b{vao_b}, src_p{&a}, dst_p {&b} {}

    CUDAGLBuffer::CUDALock lock_src() const { return src_p->lock(); }
    CUDAGLBuffer::CUDALock lock_dst() const { return dst_p->lock(); }

    void swap() {
        std::swap(src_p, dst_p);
    }

private:
    CUDAGLBuffer a, b;
    CUDAGLBuffer *src_p, *dst_p;
};

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
    ParticleData(unsigned fluid_n, unsigned boundary_n, float cell_size)
        : fluid_n{fluid_n}, boundary_n{boundary_n}, cell_size{cell_size},
        morton_codes(fluid_n),
        boundary_morton_codes(boundary_n),
        indices(fluid_n),
        boundary_indices(boundary_n),
        density_buf(fluid_n),
        boundary_mass_buf(boundary_n),
        pressure_buf(fluid_n),
        velocity_buf(fluid_n),
        non_pressure_accel_buf(fluid_n),
        normal_buf(fluid_n) {}

    void init_positions(GLuint pos_vao_a, GLuint pos_vao_b) {
        position_buf = std::make_unique<DoubleGLBuffer>(pos_vao_a, pos_vao_b);
    }

    void sort_boundary(float4 *positions_src, float4 *positions_dst) {
        assert(boundary_n > 0);

        thrust::sequence(boundary_indices.begin(), boundary_indices.end());

        const thrust::device_ptr<float4> pos_src{positions_src + fluid_n};
        const float cs = cell_size;
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

        thrust::gather(boundary_indices.begin(), boundary_indices.end(), boundary_mass_buf.src().begin(), boundary_mass_buf.dst().begin());
        boundary_mass_buf.swap();

        // Sequence again for visualization
        thrust::sequence(boundary_indices.begin(), boundary_indices.end());
    }

    void sort(const float4 *positions_src, float4 *positions_dst) {
        thrust::sequence(indices.begin(), indices.end());

        const thrust::device_ptr<const float4> pos_src{positions_src};
        const float cs = cell_size;
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
        gather_buffers();
        swap_buffers();
    }

    const DoubleGLBuffer& position() const {
        assert(position_buf != nullptr);
        return *position_buf;
    }

    float* density() { return dev_ptr(density_buf.src()); }
    float* boundary_mass() { return dev_ptr(boundary_mass_buf.src()); }
    float* pressure() { return dev_ptr(pressure_buf.src()); }
    float4* velocity() { return dev_ptr(velocity_buf.src()); }
    float4* non_pressure_accel() { return dev_ptr(non_pressure_accel_buf.src()); }
    float4* normal() { return dev_ptr(normal_buf.src()); }

    thrust::device_vector<float>& density_vec() { return density_buf.src(); }
    thrust::device_vector<float>& boundary_mass_vec() { return boundary_mass_buf.src(); }
    thrust::device_vector<float>& pressure_vec() { return pressure_buf.src(); }
    thrust::device_vector<float4>& velocity_vec() { return velocity_buf.src(); }
    thrust::device_vector<float4>& non_pressure_accel_vec() { return non_pressure_accel_buf.src(); }
    thrust::device_vector<float4>& normal_vec() { return normal_buf.src(); }

    const thrust::device_vector<float>& density_vec() const { return density_buf.src(); }
    const thrust::device_vector<float>& boundary_mass_vec() const { return boundary_mass_buf.src(); }
    const thrust::device_vector<float>& pressure_vec() const { return pressure_buf.src(); }
    const thrust::device_vector<float4>& velocity_vec() const { return velocity_buf.src(); }
    const thrust::device_vector<float4>& non_pressure_accel_vec() const { return non_pressure_accel_buf.src(); }
    const thrust::device_vector<float4>& normal_vec() const { return normal_buf.src(); }

    unsigned* get_indices() { return dev_ptr(indices); }
    unsigned* get_boundary_indices() { return dev_ptr(boundary_indices); }

    const unsigned* get_indices() const { return dev_ptr(indices); }
    const unsigned* get_boundary_indices() const { return dev_ptr(boundary_indices); }

private:
    void swap_buffers() {
        density_buf.swap();
        pressure_buf.swap();
        velocity_buf.swap();
        non_pressure_accel_buf.swap();
        normal_buf.swap();
        position_buf->swap();
    }

    void gather_buffers() {
        thrust::gather(indices.begin(), indices.end(), density_buf.src().begin(), density_buf.dst().begin());
        thrust::gather(indices.begin(), indices.end(), pressure_buf.src().begin(), pressure_buf.dst().begin());
        thrust::gather(indices.begin(), indices.end(), velocity_buf.src().begin(), velocity_buf.dst().begin());
        thrust::gather(indices.begin(), indices.end(), non_pressure_accel_buf.src().begin(), non_pressure_accel_buf.dst().begin());
        thrust::gather(indices.begin(), indices.end(), normal_buf.src().begin(), normal_buf.dst().begin());
    }

    unsigned fluid_n, boundary_n;
    float cell_size;
    thrust::device_vector<morton_t> morton_codes, boundary_morton_codes;
    thrust::device_vector<unsigned> indices, boundary_indices;

    DoubleBuffer<float> density_buf, boundary_mass_buf, pressure_buf;
    DoubleBuffer<float4> velocity_buf, non_pressure_accel_buf, normal_buf;

    std::unique_ptr<DoubleGLBuffer> position_buf;
};
