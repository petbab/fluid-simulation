#pragma once
#include <glm/glm.hpp>

#include "ubo.h"


class Material {
    struct MaterialData {
        glm::vec3 ambient;
        float _pad0{};
        glm::vec3 diffuse;
        float alpha;
        glm::vec3 specular;
        float shininess;
    };

public:
    Material(const glm::vec3 &ambient, const glm::vec3 &diffuse, const glm::vec3 &specular,
             float shininess, float alpha) : data{ambient, 0., diffuse, alpha, specular, shininess} {
        ubo.upload_data(&data);
    }
    Material() : Material(glm::vec3{0.5f}, glm::vec3{0.5f}, glm::vec3{0.5f}, 0.f, 1.f) {}

    void set(const glm::vec3 &ambient, const glm::vec3 &diffuse, const glm::vec3 &specular,
             float shininess, float alpha) {
        data = {ambient, 0., diffuse, alpha, specular, shininess};
        ubo.upload_data(&data);
    }

    void bind_ubo(unsigned binding) const { ubo.bind(binding); }

private:
    UBO<MaterialData> ubo;
    MaterialData data;
};
