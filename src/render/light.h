#pragma once

#include <glm/glm.hpp>

#include "ubo.h"


/**
 * @brief Manages an array of Phong lights stored in a UBO.
 *
 * Supports up to MAX_LIGHTS point and directional lights, plus a global ambient term.
 * The data layout matches the PhongLightsBuffer UBO expected by shaders.
 */
class LightArray {
public:
    static unsigned constexpr MAX_LIGHTS = 8;  ///< Maximum number of lights.

private:
    /**
     * @brief Data for a single Phong light.
     *
     * Matches the std140 layout used in shaders.
     */
    struct LightData {
        LightData() = default;
        /**
         * @brief Constructs a LightData entry.
         * @param position Light position (w=1 for point, w=0 for directional).
         * @param ambient Ambient color.
         * @param diffuse Diffuse color.
         * @param specular Specular color.
         * @param atten_constant Constant attenuation.
         * @param atten_linear Linear attenuation.
         * @param atten_quadratic Quadratic attenuation.
         */
        LightData(const glm::vec4& position, const glm::vec3& ambient,
                  const glm::vec3& diffuse, const glm::vec3& specular,
                  float atten_constant, float atten_linear, float atten_quadratic)
            : position(position), ambient(ambient),
              diffuse(diffuse), specular(specular),
              atten_constant(atten_constant),
              atten_linear(atten_linear),
              atten_quadratic(atten_quadratic) {
        }

        glm::vec4 position;       ///< Position (w=1 point, w=0 directional).
        glm::vec3 ambient;        ///< Ambient color.
        float _pad0{};            ///< Padding for std140 alignment.
        glm::vec3 diffuse;        ///< Diffuse color.
        float _pad1{};            ///< Padding for std140 alignment.
        glm::vec3 specular;       ///< Specular color.
        float atten_constant;     ///< Constant attenuation.
        float atten_linear;       ///< Linear attenuation.
        float atten_quadratic;    ///< Quadratic attenuation.
        float _pad2{};            ///< Padding for std140 alignment.
        float _pad3{};            ///< Padding for std140 alignment.
    };

    /** @brief UBO data layout matching the shader's PhongLightsBuffer. */
    struct LightsBufferData {
        glm::vec3 global_ambient_color;  ///< Global ambient light color.
        int lights_count;                ///< Number of active lights.
        LightData lights[MAX_LIGHTS];    ///< Array of light data.
    };

    /*
struct PhongLight
{
    vec4 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float atten_constant;
    float atten_linear;
    float atten_quadratic;
};

layout (std140, binding = 2) uniform PhongLightsBuffer
{
    vec3 global_ambient_color;
    int lights_count;
    PhongLight lights[8];
};
     */

public:
    LightArray() : ubo{}, data{glm::vec3{0.f}, 0} {
    }

    /**
     * @brief Adds a point light with default attenuation (1, 0, 0).
     * @param position World-space position.
     * @param ambient Ambient color.
     * @param diffuse Diffuse color.
     * @param specular Specular color.
     */
    void add_point_light(const glm::vec3& position, const glm::vec3& ambient,
                         const glm::vec3& diffuse, const glm::vec3& specular) {
        add_point_light(position, ambient, diffuse, specular, 1.0f, 0.0f, 0.0f);
    }

    /**
     * @brief Adds a point light with custom attenuation.
     * @param position World-space position.
     * @param ambient Ambient color.
     * @param diffuse Diffuse color.
     * @param specular Specular color.
     * @param atten_constant Constant attenuation.
     * @param atten_linear Linear attenuation.
     * @param atten_quadratic Quadratic attenuation.
     */
    void add_point_light(const glm::vec3& position, const glm::vec3& ambient, const glm::vec3& diffuse,
                         const glm::vec3& specular, float atten_constant, float atten_linear,
                         float atten_quadratic) {
        add_light({
            glm::vec4(position, 1.0f), ambient, diffuse, specular,
            atten_constant, atten_linear, atten_quadratic
        });
    }

    /**
     * @brief Adds a directional light.
     * @param position Light direction.
     * @param ambient Ambient color.
     * @param diffuse Diffuse color.
     * @param specular Specular color.
     */
    void add_directional_light(const glm::vec3& position, const glm::vec3& ambient,
                               const glm::vec3& diffuse, const glm::vec3& specular) {
        add_light({
            glm::vec4(position, 0.0f), ambient, diffuse, specular, 1.0f, 0.0f, 0.0f
        });
    }

    /**
     * @brief Sets the global ambient light color.
     * @param ambient RGB ambient color.
     */
    void set_ambient_light(const glm::vec3& ambient) {
        data.global_ambient_color = ambient;
    }

    /** @brief Uploads the current light data to the GPU UBO. */
    void upload_data() { ubo.upload_data(&data); }

    /**
     * @brief Binds the light UBO to the given binding point.
     * @param binding UBO binding index.
     */
    void bind_ubo(unsigned binding) const { ubo.bind(binding); }

private:
    void add_light(LightData l) {
        if (data.lights_count < MAX_LIGHTS)
            data.lights[data.lights_count++] = std::move(l);
    }

    UBO<LightsBufferData> ubo;  ///< Uniform buffer object for light data.
    LightsBufferData data;      ///< CPU-side light buffer data.
};
