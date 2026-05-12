#pragma once

#include "object.h"


/**
 * @brief Represents an axis-aligned bounding box with a transform.
 */
struct BoundingBox {
    /**
     * @brief Constructs a bounding box.
     * @param min Minimum corner in local space.
     * @param max Maximum corner in local space.
     */
    BoundingBox(glm::vec3 min, glm::vec3 max) : min{min}, max{max}, model{1.f}, model_inv{1.f} {}

    glm::vec3 min, max;   ///< Minimum and maximum corners in local space.
    glm::mat4 model;      ///< Model matrix transforming the box to world space.
    glm::mat4 model_inv;  ///< Inverse of the model matrix.
};

/**
 * @brief Renderable box object with a bounding box.
 *
 * Uses front-face culling during rendering to create an inside-out box effect,
 * suitable for rendering container boundaries.
 */
class Box : public Object {
public:
    /**
     * @brief Constructs a box.
     * @param half_size Half-extents of the box in each axis.
     * @param color RGB color of the box.
     */
    Box(glm::vec3 half_size, glm::vec3 color);

    /**
     * @brief Returns the bounding box of this box.
     * @return Const reference to the BoundingBox.
     */
    const BoundingBox& bounding_box() const { return *bb; }

    void render() const override;

    /**
     * @brief Sets the model matrix and updates the bounding box transform.
     * @param m New model matrix.
     */
    void set_model(const glm::mat4& m);

protected:
    std::unique_ptr<BoundingBox> bb;  ///< The bounding box geometry.
};
