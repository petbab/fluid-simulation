#pragma once

#include "shader.h"
#include "geometry.h"


class Object {
public:
    Object(const Shader &shader, const Geometry &geometry);

    void render() const;

private:
    const Shader &shader;
    const Geometry &geometry;
};
