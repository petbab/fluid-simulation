#pragma once

#include "cuda_gl_buffer.h"
#include "../render/ssbo.h"
#include "../render/shader.h"
#include <type_traits>

namespace detail_ {

template<typename T>
class TypeVisualizer {
public:
    explicit TypeVisualizer(unsigned total_particles, unsigned fluid_particles) : ssbo{total_particles},
        cuda_gl_buffer{ssbo.get_id(), cudaGraphicsRegisterFlagsWriteDiscard},
        total_particles{total_particles}, fluid_particles{fluid_particles} {}

    void visualize(Shader* shader, const T* data, bool visualize_boundary) {
        shader->use();

        if constexpr (std::is_same_v<T, float>) {
            shader->set_uniform("visualize_float", true);
            ssbo.bind(SSBO<void>::VISUALIZE_FLOAT_SSBO_BINDING);
        } else {
            shader->set_uniform("visualize_vec", true);
            ssbo.bind(SSBO<void>::VISUALIZE_VEC3_SSBO_BINDING);
        }

        shader->set_uniform("visualize_boundary", visualize_boundary);

        write_data(data, visualize_boundary ? total_particles - fluid_particles : fluid_particles);
    }

    void visualize(Shader* shader, const T* data, float min_value, float max_value, bool visualize_boundary) {
        visualize(shader, data, visualize_boundary);
        shader->set_uniform("norm", true);
        shader->set_uniform("min_value", min_value);
        shader->set_uniform("max_value", max_value);
    }

private:
    void write_data(const T *src, unsigned particles) {
        CUDAGLBuffer::CUDALock l = cuda_gl_buffer.lock();
        T *dst = static_cast<T*>(l.get_ptr());
        cudaMemcpy(dst, src, sizeof(T) * particles, cudaMemcpyDeviceToDevice);
        cudaCheckError();
    }

    SSBO<T> ssbo;
    CUDAGLBuffer cuda_gl_buffer;
    unsigned total_particles, fluid_particles;
};

}


class ParticleDataVisualizer {
public:
    explicit ParticleDataVisualizer(unsigned total_particles, unsigned fluid_particles)
        : float_visualizer{total_particles, fluid_particles}, vec_visualizer{total_particles, fluid_particles} {}

    void visualize(Shader* shader, const float* data, bool visualize_boundary = false) {
        float_visualizer.visualize(shader, data, visualize_boundary);
    }
    void visualize(Shader* shader, const float* data, float min_value, float max_value, bool visualize_boundary = false) {
        float_visualizer.visualize(shader, data, min_value, max_value, visualize_boundary);
    }
    void visualize(Shader* shader, const float4* data, bool visualize_boundary = false) {
        vec_visualizer.visualize(shader, data, visualize_boundary);
    }
    void visualize(Shader* shader, const float4* data, float min_value, float max_value, bool visualize_boundary = false) {
        vec_visualizer.visualize(shader, data, min_value, max_value, visualize_boundary);
    }

private:
    detail_::TypeVisualizer<float> float_visualizer;
    detail_::TypeVisualizer<float4> vec_visualizer;
};
