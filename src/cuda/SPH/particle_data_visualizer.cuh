#pragma once

#include "../cuda_gl_buffer.h"
#include <render/ssbo.h>
#include <render/shader.h>
#include <type_traits>
#include <map>
#include "particle_data.cuh"


namespace detail_ {

/**
 * @brief Typed SSBO visualizer for particle data.
 *
 * Manages an SSBO and CUDA-GL interop buffer for a specific data type
 * (float, unsigned, or float4), and configures shader uniforms accordingly.
 * @tparam T Data type stored in the SSBO.
 */
template<typename T>
class TypeVisualizer {
public:
    /**
     * @brief Constructs the type visualizer.
     * @param total_particles Total number of particles (fluid + boundary).
     * @param fluid_particles Number of fluid particles.
     */
    explicit TypeVisualizer(unsigned total_particles, unsigned fluid_particles) : ssbo{total_particles},
        cuda_gl_buffer{ssbo.get_id(), cudaGraphicsRegisterFlagsWriteDiscard},
        total_particles{total_particles}, fluid_particles{fluid_particles} {}

    /**
     * @brief Configures shader uniforms for visualization.
     * @param shader Shader to configure.
     * @param visualize_boundary If true, visualizes boundary particles.
     */
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

    /**
     * @brief Configures shader uniforms with normalization.
     * @param shader Shader to configure.
     * @param visualize_boundary If true, visualizes boundary particles.
     * @param min_value Minimum value for normalization.
     * @param max_value Maximum value for normalization.
     */
    void visualize(Shader* shader, bool visualize_boundary, float min_value, float max_value) {
        visualize(shader, visualize_boundary);
        shader->set_uniform("norm", true);
        shader->set_uniform("min_value", min_value);
        shader->set_uniform("max_value", max_value);
    }

    /**
     * @brief Copies particle data from device to the SSBO.
     * @param data Device pointer to source data.
     * @param boundary If true, copies boundary particles; otherwise fluid particles.
     */
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

    SSBO<T> ssbo;                ///< Shader storage buffer object.
    CUDAGLBuffer cuda_gl_buffer; ///< CUDA-GL interop buffer.
    unsigned total_particles, fluid_particles;  ///< Particle counts.
};

}


/**
 * @brief Visualizes particle data by mapping it to SSBOs for shader rendering.
 *
 * Supports multiple visualization modes (density, pressure, velocity, etc.)
 * and automatically configures shader uniforms.
 */
class ParticleDataVisualizer {
    template <class T>
    static const T* dev_ptr(const thrust::device_vector<T>& vec) {
        return thrust::raw_pointer_cast(vec.data());
    }

public:
    /**
     * @brief Constructs the visualizer.
     * @param data Pointer to particle data.
     * @param total_particles Total number of particles.
     * @param fluid_particles Number of fluid particles.
     */
    explicit ParticleDataVisualizer(const ParticleData *data, unsigned total_particles, unsigned fluid_particles)
        : particle_data(data), total_particles(total_particles), fluid_particles(fluid_particles),
        float_visualizer{total_particles, fluid_particles},
        uint_visualizer{total_particles, fluid_particles},
        vec_visualizer{total_particles, fluid_particles} {}

    /** @brief Available visualization modes. */
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

    /** @brief Mode specification (normalization range). */
    struct mode_spec_t {
        bool normalize;  ///< Whether to normalize values.
        float min, max;  ///< Normalization range.
    };

    static const std::map<mode_t, mode_spec_t> modes;      ///< Mode specifications.
    static const std::vector<const char*> mode_strings;    ///< Human-readable mode names.

    /** @brief Updates SSBO data based on the current visualization mode. */
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

    /**
     * @brief Configures shader uniforms for the current visualization mode.
     * @param shader Shader to configure.
     */
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

    /**
     * @brief Sets the visualization mode and updates data.
     * @param m New mode.
     */
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
    mode_t mode = mode_t::none;  ///< Current visualization mode.
    bool normalize = false;      ///< Whether normalization is active.
    float min = 0.f, max = 0.f;  ///< Current normalization range.

private:
    const ParticleData *particle_data;  ///< Source particle data.
    unsigned total_particles, fluid_particles;  ///< Particle counts.
    detail_::TypeVisualizer<float> float_visualizer;     ///< Float data visualizer.
    detail_::TypeVisualizer<unsigned> uint_visualizer;   ///< Unsigned data visualizer.
    detail_::TypeVisualizer<float4> vec_visualizer;      ///< Vector data visualizer.
};
