#pragma once

#include "shader.h"
#include "geometry.h"


struct BoundingBox {
    glm::vec3 min, max;
};

class Object {
public:
    Object(Shader *shader, Geometry *geometry);
    virtual ~Object() = default;

    virtual void render() const;
    virtual void update(double delta);

    void set_model(const glm::mat4 &m);

protected:
    Object() = default;

    Shader *shader = nullptr;
    Geometry *geometry = nullptr;
};
