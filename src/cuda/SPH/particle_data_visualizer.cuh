#pragma once

#include "../cuda_gl_buffer.h"
#include <render/ssbo.h>
#include <render/shader.h>
#include <type_traits>
#include <map>
#include "particle_data.cuh"


namespace detail_ {

template<typename T>
class TypeVisualizer {
public:
    explicit TypeVisualizer(unsigned total_particles, unsigned fluid_particles) : ssbo{total_particles},
        cuda_gl_buffer{ssbo.get_id(), cudaGraphicsRegisterFlagsWriteDiscard},
        total_particles{total_particles}, fluid_particles{fluid_particles} {}

    void visualize(Shader* shader, bool visualize_boundary) {
        shader->use();

        if constexpr (std::is_same_v<T, float>) {
            shader->set_uniform("visualize_float", true);
            shader->set_uniform("visualize_vec", false);
            shader->set_uniform("visualize_uint", false);
            ssbo.bind(SSBO<void>::VISUALIZE_FLOAT_SSBO_BINDING);
        } else if constexpr (std::is_same_v<T, float4>) {
            shader->set_uniform("visualize_float", false);
            shader->set_uniform("visualize_vec", true);
            shader->set_uniform("visualize_uint", false);
            ssbo.bind(SSBO<void>::VISUALIZE_VEC3_SSBO_BINDING);
        } else {
            shader->set_uniform("visualize_float", false);
            shader->set_uniform("visualize_vec", false);
            shader->set_uniform("visualize_uint", true);
            ssbo.bind(SSBO<void>::VISUALIZE_UINT_SSBO_BINDING);
        }

        shader->set_uniform("visualize_boundary", visualize_boundary);
    }

    void visualize(Shader* shader, bool visualize_boundary, float min_value, float max_value) {
        visualize(shader, visualize_boundary);
        shader->set_uniform("norm", true);
        shader->set_uniform("min_value", min_value);
        shader->set_uniform("max_value", max_value);
    }

    void update(const T* data, bool boundary) {
        write_data(data, boundary ? total_particles - fluid_particles : fluid_particles);
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
    template <class T>
    static const T* dev_ptr(const thrust::device_vector<T>& vec) {
        return thrust::raw_pointer_cast(vec.data());
    }

public:
    explicit ParticleDataVisualizer(const ParticleData *data, unsigned total_particles, unsigned fluid_particles)
        : particle_data(data), total_particles(total_particles), fluid_particles(fluid_particles),
        float_visualizer{total_particles, fluid_particles},
        uint_visualizer{total_particles, fluid_particles},
        vec_visualizer{total_particles, fluid_particles} {}

    enum class mode_t {
        pretty,
        none,
        density,
        pressure,
        velocity,
        non_pressure_accel,
        normal,
        indices,
        boundary_mass,
        boundary_indices,
    };

    struct mode_spec_t {
        bool normalize;
        float min, max;
    };

    static const std::map<mode_t, mode_spec_t> modes;
    static const std::vector<const char*> mode_strings;

    void update() {
        switch (mode) {
        case mode_t::pretty:
            // TODO
            break;
        case mode_t::none:
            break;
        case mode_t::density:
            float_visualizer.update(dev_ptr(particle_data->density_vec()), false);
            break;
        case mode_t::pressure:
            float_visualizer.update(dev_ptr(particle_data->pressure_vec()), false);
            break;
        case mode_t::velocity:
            vec_visualizer.update(dev_ptr(particle_data->velocity_vec()), false);
            break;
        case mode_t::non_pressure_accel:
            vec_visualizer.update(dev_ptr(particle_data->non_pressure_accel_vec()), false);
            break;
        case mode_t::normal:
            vec_visualizer.update(dev_ptr(particle_data->normal_vec()), false);
            break;
        case mode_t::indices:
            uint_visualizer.update(particle_data->get_indices(), false);
            break;
        case mode_t::boundary_mass:
            if (has_boundary())
                float_visualizer.update(dev_ptr(particle_data->boundary_mass_vec()), true);
            break;
        case mode_t::boundary_indices:
            if (has_boundary())
                uint_visualizer.update(particle_data->get_boundary_indices(), true);
            break;
        }
    }

    void visualize(Shader* shader) {
        switch (mode) {
        case mode_t::pretty:
            // TODO
            break;
        case mode_t::none:
            shader->set_uniform("visualize_float", false);
            shader->set_uniform("visualize_vec", false);
            shader->set_uniform("visualize_uint", false);
            break;
        case mode_t::density:
        case mode_t::pressure:
            float_visualizer.visualize(shader, false, min, max);
            break;
        case mode_t::velocity:
        case mode_t::non_pressure_accel:
        case mode_t::normal:
            vec_visualizer.visualize(shader, false, min, max);
            break;
        case mode_t::indices:
            uint_visualizer.visualize(shader, false, min, max);
            break;
        case mode_t::boundary_mass:
            if (has_boundary())
                float_visualizer.visualize(shader, true, min, max);
            break;
        case mode_t::boundary_indices:
            if (has_boundary())
                uint_visualizer.visualize(shader, true, min, max);
            break;
        }
    }

    void set_mode(mode_t m) {
        mode = m;
        const auto &spec = modes.at(m);
        switch (m) {
        case mode_t::indices:
            min = 0.f;
            max = static_cast<float>(fluid_particles);
            break;
        case mode_t::boundary_indices:
            if (has_boundary()) {
                min = 0.f;
                max = static_cast<float>(total_particles - fluid_particles);
            }
            break;
        default:
            min = spec.min;
            max = spec.max;
            break;
        }
        normalize = spec.normalize;
        update();
    }

private:
    bool has_boundary() const { return total_particles > fluid_particles; }

public:
    mode_t mode = mode_t::none;
    bool normalize = false;
    float min = 0.f, max = 0.f;

private:
    const ParticleData *particle_data;
    unsigned total_particles, fluid_particles;
    detail_::TypeVisualizer<float> float_visualizer;
    detail_::TypeVisualizer<unsigned> uint_visualizer;
    detail_::TypeVisualizer<float4> vec_visualizer;
};
