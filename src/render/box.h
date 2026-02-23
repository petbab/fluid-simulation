#pragma once

#include "object.h"


class Box : public Object {
public:
    Box(glm::vec3 half_size, glm::vec3 color);

    const BoundingBox& bounding_box() const { return *bb; }

    void render() const override;

    void set_model(const glm::mat4& m);

protected:
    std::unique_ptr<BoundingBox> bb;
};
