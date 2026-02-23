#pragma once

#include "object.h"


struct BoundingBox {
    BoundingBox(glm::vec3 min, glm::vec3 max) : min{min}, max{max}, model{1.f}, model_inv{1.f} {}

    glm::vec3 min, max;
    glm::mat4 model, model_inv;
};

class Box : public Object {
public:
    Box(glm::vec3 half_size, glm::vec3 color);

    const BoundingBox& bounding_box() const { return *bb; }

    void render() const override;

    void set_model(const glm::mat4& m);

protected:
    std::unique_ptr<BoundingBox> bb;
};
