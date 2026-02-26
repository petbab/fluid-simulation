#pragma once

#include <glm/glm.hpp>

#include "ubo.h"


class LightArray {
public:
    static unsigned constexpr MAX_LIGHTS = 8;

private:
    struct LightData {
        LightData() = default;
        LightData(const glm::vec4& position, const glm::vec3& ambient,
                  const glm::vec3& diffuse, const glm::vec3& specular,
                  float atten_constant, float atten_linear, float atten_quadratic)
            : position(position), ambient(ambient),
              diffuse(diffuse), specular(specular),
              atten_constant(atten_constant),
              atten_linear(atten_linear),
              atten_quadratic(atten_quadratic) {
        }

        glm::vec4 position;
        glm::vec3 ambient;
        float _pad0{};
        glm::vec3 diffuse;
        float _pad1{};
        glm::vec3 specular;
        float atten_constant;
        float atten_linear;
        float atten_quadratic;
        float _pad2{};
        float _pad3{};
    };

    struct LightsBufferData {
        glm::vec3 global_ambient_color;
        int lights_count;
        LightData lights[MAX_LIGHTS];
    };

    /*
// The structure holding the information about a single Phong light.
struct PhongLight
{
    vec4 position;                   // The position of the light. Note that position.w should be one for point lights, and zero for directional lights.
    vec3 ambient;                    // The ambient part of the color of the light.
    vec3 diffuse;                    // The diffuse part of the color of the light.
    vec3 specular;                   // The specular part of the color of the light.
    float atten_constant;            // The constant attenuation of point lights, irrelevant for directional lights. For no attenuation, set this to 1.
    float atten_linear;              // The linear attenuation of point lights, irrelevant for directional lights.  For no attenuation, set this to 0.
    float atten_quadratic;           // The quadratic attenuation of point lights, irrelevant for directional lights. For no attenuation, set this to 0.
};

// The UBO with light data.
layout (std140, binding = 2) uniform PhongLightsBuffer
{
    vec3 global_ambient_color;		 // The global ambient color.
    int lights_count;				 // The number of lights in the buffer.
    PhongLight lights[8];			 // The array with actual lights.
};
     */

public:
    LightArray() : ubo{}, data{glm::vec3{0.f}, 0} {
    }

    void add_point_light(const glm::vec3& position, const glm::vec3& ambient,
                         const glm::vec3& diffuse, const glm::vec3& specular) {
        add_point_light(position, ambient, diffuse, specular, 1.0f, 0.0f, 0.0f);
    }

    void add_point_light(const glm::vec3& position, const glm::vec3& ambient, const glm::vec3& diffuse,
                         const glm::vec3& specular, float atten_constant, float atten_linear,
                         float atten_quadratic) {
        add_light({
            glm::vec4(position, 1.0f), ambient, diffuse, specular,
            atten_constant, atten_linear, atten_quadratic
        });
    }

    void add_directional_light(const glm::vec3& position, const glm::vec3& ambient,
                               const glm::vec3& diffuse, const glm::vec3& specular) {
        add_light({
            glm::vec4(position, 0.0f), ambient, diffuse, specular, 1.0f, 0.0f, 0.0f
        });
    }

    void set_ambient_light(const glm::vec3& ambient) {
        data.global_ambient_color = ambient;
    }

    void upload_data() { ubo.upload_data(&data); }
    void bind_ubo(unsigned binding) const { ubo.bind(binding); }

private:
    void add_light(LightData l) {
        if (data.lights_count < MAX_LIGHTS)
            data.lights[data.lights_count++] = std::move(l);
    }

    UBO<LightsBufferData> ubo;
    LightsBufferData data;
};
