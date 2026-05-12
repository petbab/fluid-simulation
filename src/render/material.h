#pragma once
#include <glm/glm.hpp>

#include "ubo.h"


/**
 * @brief Phong material properties stored in a UBO.
 *
 * The data layout matches the Material UBO expected by lit shaders.
 */
class Material {
    /** @brief std140-aligned material data. */
    struct MaterialData {
        glm::vec3 ambient;   ///< Ambient reflectivity.
        float _pad0{};       ///< Padding for std140 alignment.
        glm::vec3 diffuse;   ///< Diffuse reflectivity.
        float alpha;         ///< Opacity (1.0 = fully opaque).
        glm::vec3 specular;  ///< Specular reflectivity.
        float shininess;     ///< Specular exponent.
    };

public:
    /**
     * @brief Constructs a material with given Phong parameters.
     * @param ambient Ambient color.
     * @param diffuse Diffuse color.
     * @param specular Specular color.
     * @param shininess Specular shininess exponent.
     * @param alpha Opacity.
     */
    Material(const glm::vec3 &ambient, const glm::vec3 &diffuse, const glm::vec3 &specular,
             float shininess, float alpha) : data{ambient, 0., diffuse, alpha, specular, shininess} {
        ubo.upload_data(&data);
    }

    /** @brief Constructs a default gray opaque material. */
    Material() : Material(glm::vec3{0.5f}, glm::vec3{0.5f}, glm::vec3{0.5f}, 0.f, 1.f) {}

    /**
     * @brief Updates material properties and uploads to GPU.
     * @param ambient Ambient color.
     * @param diffuse Diffuse color.
     * @param specular Specular color.
     * @param shininess Specular shininess exponent.
     * @param alpha Opacity.
     */
    void set(const glm::vec3 &ambient, const glm::vec3 &diffuse, const glm::vec3 &specular,
             float shininess, float alpha) {
        data = {ambient, 0., diffuse, alpha, specular, shininess};
        ubo.upload_data(&data);
    }

    /**
     * @brief Binds the material UBO to the given binding point.
     * @param binding UBO binding index.
     */
    void bind_ubo(unsigned binding) const { ubo.bind(binding); }

private:
    UBO<MaterialData> ubo;  ///< Uniform buffer object for material data.
    MaterialData data;      ///< CPU-side material data.
};
