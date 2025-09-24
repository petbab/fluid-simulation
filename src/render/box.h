#pragma once

#include "object.h"


class Box : public Object {
public:
    Box(glm::vec3 min, glm::vec3 max, glm::vec4 color);
    Box(BoundingBox bb, glm::vec4 color) : Box{bb.min, bb.max, color} {}

    const BoundingBox& bounding_box() const { return *bb; }

    void render() const override;

protected:
    std::unique_ptr<BoundingBox> bb;
};

class MovingBox : public Box {
public:
    MovingBox(glm::vec3 min, glm::vec3 max, glm::vec4 color);

    void update(float delta) override;

private:
    float time = 0.;
    const BoundingBox initial_bb;
};
