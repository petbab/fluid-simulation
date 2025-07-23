#pragma once

#include "shader.h"
#include "geometry.h"


struct BoundingBox {
    glm::vec3 min, max;
};

class Object {
public:
    Object(const Shader *shader, const Geometry *geometry);
    virtual ~Object() = default;

    virtual void render() const;
    virtual void update(double delta);

protected:
    Object() = default;

    const Shader *shader = nullptr;
    const Geometry *geometry = nullptr;
};
