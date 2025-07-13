#pragma once

#include "shader.h"
#include "geometry.h"


class Object {
public:
    Object(const Shader &shader, const Geometry &geometry);
    virtual ~Object() = default;

    virtual void render() const;
    virtual void update(double delta);

protected:
    const Shader &shader;
    const Geometry &geometry;
};
