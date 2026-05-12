#pragma once

#include "shader.h"
#include "geometry.h"
#include "material.h"
#include "model.h"


/**
 * @brief Base class for renderable objects.
 *
 * Combines a shader, geometry, material, and model transform.
 * Derived classes can override render() and update() for custom behavior.
 */
class Object {
public:
    /**
     * @brief Constructs an object.
     * @param shader Shader to use for rendering.
     * @param geometry Geometry to render.
     */
    Object(Shader* shader, Geometry* geometry);
    virtual ~Object() = default;

    /**
     * @brief Renders the object.
     *
     * Binds material and model UBOs, activates the shader, and draws the geometry.
     */
    virtual void render() const;

    /**
     * @brief Updates the object state.
     * @param delta Time step in seconds.
     */
    virtual void update(float delta) {}

    /**
     * @brief Sets the model transformation matrix.
     * @param m New model matrix.
     */
    void set_model(const glm::mat4& m) { model.set(m); }

    /**
     * @brief Sets the material properties.
     * @param ambient Ambient color.
     * @param diffuse Diffuse color.
     * @param specular Specular color.
     * @param shininess Specular shininess.
     * @param alpha Opacity.
     */
    void set_material(const glm::vec3& ambient, const glm::vec3& diffuse,
                      const glm::vec3& specular, float shininess, float alpha) {
        material.set(ambient, diffuse, specular, shininess, alpha);
    }

    /** @return Pointer to the geometry. */
    const Geometry *get_geometry() const { return geometry; }
    /** @return The current model matrix. */
    const glm::mat4& get_model() const { return model.get_model(); }

protected:
    Object() = default;

    Shader* shader = nullptr;      ///< Shader used for rendering.
    Geometry* geometry = nullptr;  ///< Geometry to render.
    Model model{};                 ///< Model transformation.
    Material material{};           ///< Material properties.
};
