#pragma once

#include "cuda_gl_buffer.h"
#include "../render/ssbo.h"
#include "../render/shader.h"
#include <type_traits>


template<typename T>
class ParticleDataVisualizer {
public:
    ParticleDataVisualizer(unsigned particles) : ssbo{particles},
        cuda_gl_buffer{ssbo.get_id(), cudaGraphicsRegisterFlagsWriteDiscard},
        particles{particles} {}

    void visualize(Shader* shader, const T* data) {
        shader->use();

        if constexpr (std::is_same_v<T, float>) {
            shader->set_uniform("visualize_float", true);
            ssbo.bind(SSBO<void>::VISUALIZE_FLOAT_SSBO_BINDING);
        } else {
            shader->set_uniform("visualize_vec", true);
            ssbo.bind(SSBO<void>::VISUALIZE_VEC3_SSBO_BINDING);
        }

        write_data(data);
    }

    void visualize(Shader* shader, const T* data, float min_value, float max_value) {
        visualize(shader, data);
        shader->set_uniform("norm", true);
        shader->set_uniform("min_value", min_value);
        shader->set_uniform("max_value", max_value);
    }

private:
    void write_data(const T *src) {
        CUDAGLBuffer::CUDALock l = cuda_gl_buffer.lock();
        T *dst = static_cast<T*>(l.get_ptr());
        cudaMemcpy(dst, src, sizeof(T) * particles, cudaMemcpyDeviceToDevice);
        cudaCheckError();
    }

    SSBO<T> ssbo;
    CUDAGLBuffer cuda_gl_buffer;
    unsigned particles;
};
