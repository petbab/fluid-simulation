#pragma once

#include "shader.h"
#include "geometry.h"
#include "material.h"
#include "model.h"


struct BoundingBox {
    glm::vec3 min, max;
};

class Object {
public:
    Object(Shader* shader, Geometry* geometry);
    virtual ~Object() = default;

    virtual void render() const;

    virtual void update(float delta) {}

    void set_model(const glm::mat4& m) { model.set(m); }
    void set_material(const glm::vec3& ambient, const glm::vec3& diffuse,
                      const glm::vec3& specular, float shininess, float alpha) {
        material.set(ambient, diffuse, specular, shininess, alpha);
    }

protected:
    Object() = default;

    Shader* shader = nullptr;
    Geometry* geometry = nullptr;
    Model model{};
    Material material{};
};
