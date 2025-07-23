#pragma once

#include "object.h"


class Box : public Object {
public:
    Box(glm::vec3 min, glm::vec3 max, glm::vec4 color);
    Box(BoundingBox bb, glm::vec4 color) : Box{bb.min, bb.max, color} {}

    operator BoundingBox() const { return {min, max}; }

    void render() const override;

private:
    glm::vec3 min, max;
};
