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

    const Geometry *get_geometry() const { return geometry; }
    const glm::mat4& get_model() const { return model.get_model(); }

protected:
    Object() = default;

    Shader* shader = nullptr;
    Geometry* geometry = nullptr;
    Model model{};
    Material material{};
};
