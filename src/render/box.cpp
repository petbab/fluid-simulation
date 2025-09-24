#include "box.h"
#include <glm/gtx/transform.hpp>
#include "asset_manager.h"
#include "../config.h"


static Geometry box_geometry(glm::vec3 half_size) {
    // Box vertices with counter-clockwise winding for OpenGL
    // Each face consists of 2 triangles (6 vertices per face, 36 total)
    // Coordinates range from -0.5 to 0.5 for a unit cube

    float positions[] = {
        // Front face (z = 0.5)
        -half_size.x, -half_size.y,  half_size.z,  // Bottom-left
        half_size.x, -half_size.y,  half_size.z,  // Bottom-right
        half_size.x,  half_size.y,  half_size.z,  // Top-right

        -half_size.x, -half_size.y,  half_size.z,  // Bottom-left
        half_size.x,  half_size.y,  half_size.z,  // Top-right
        -half_size.x,  half_size.y,  half_size.z,  // Top-left

        // Back face (z = -0.5)
        half_size.x, -half_size.y, -half_size.z,  // Bottom-right
        -half_size.x, -half_size.y, -half_size.z,  // Bottom-left
        -half_size.x,  half_size.y, -half_size.z,  // Top-left

        half_size.x, -half_size.y, -half_size.z,  // Bottom-right
        -half_size.x,  half_size.y, -half_size.z,  // Top-left
        half_size.x,  half_size.y, -half_size.z,  // Top-right

        // Left face (x = -0.5)
        -half_size.x, -half_size.y, -half_size.z,  // Bottom-back
        -half_size.x, -half_size.y,  half_size.z,  // Bottom-front
        -half_size.x,  half_size.y,  half_size.z,  // Top-front

        -half_size.x, -half_size.y, -half_size.z,  // Bottom-back
        -half_size.x,  half_size.y,  half_size.z,  // Top-front
        -half_size.x,  half_size.y, -half_size.z,  // Top-back

        // Right face (x = 0.5)
        half_size.x, -half_size.y,  half_size.z,  // Bottom-front
        half_size.x, -half_size.y, -half_size.z,  // Bottom-back
        half_size.x,  half_size.y, -half_size.z,  // Top-back

        half_size.x, -half_size.y,  half_size.z,  // Bottom-front
        half_size.x,  half_size.y, -half_size.z,  // Top-back
        half_size.x,  half_size.y,  half_size.z,  // Top-front

        // Top face (y = 0.5)
        -half_size.x,  half_size.y,  half_size.z,  // Front-left
        half_size.x,  half_size.y,  half_size.z,  // Front-right
        half_size.x,  half_size.y, -half_size.z,  // Back-right

        -half_size.x,  half_size.y,  half_size.z,  // Front-left
        half_size.x,  half_size.y, -half_size.z,  // Back-right
        -half_size.x,  half_size.y, -half_size.z,  // Back-left

        // Bottom face (y = -0.5)
        -half_size.x, -half_size.y, -half_size.z,  // Back-left
        half_size.x, -half_size.y, -half_size.z,  // Back-right
        half_size.x, -half_size.y,  half_size.z,  // Front-right

        -half_size.x, -half_size.y, -half_size.z,  // Back-left
        half_size.x, -half_size.y,  half_size.z,  // Front-right
        -half_size.x, -half_size.y,  half_size.z   // Front-left
    };

    // Optional: Normals for each vertex (same order as vertices)
    float normals[] = {
        // Front face normals
        0.0f,  0.0f,  1.0f,
        0.0f,  0.0f,  1.0f,
        0.0f,  0.0f,  1.0f,
        0.0f,  0.0f,  1.0f,
        0.0f,  0.0f,  1.0f,
        0.0f,  0.0f,  1.0f,

        // Back face normals
        0.0f,  0.0f, -1.0f,
        0.0f,  0.0f, -1.0f,
        0.0f,  0.0f, -1.0f,
        0.0f,  0.0f, -1.0f,
        0.0f,  0.0f, -1.0f,
        0.0f,  0.0f, -1.0f,

        // Left face normals
        -1.0f,  0.0f,  0.0f,
        -1.0f,  0.0f,  0.0f,
        -1.0f,  0.0f,  0.0f,
        -1.0f,  0.0f,  0.0f,
        -1.0f,  0.0f,  0.0f,
        -1.0f,  0.0f,  0.0f,

        // Right face normals
        1.0f,  0.0f,  0.0f,
        1.0f,  0.0f,  0.0f,
        1.0f,  0.0f,  0.0f,
        1.0f,  0.0f,  0.0f,
        1.0f,  0.0f,  0.0f,
        1.0f,  0.0f,  0.0f,

        // Top face normals
        0.0f,  1.0f,  0.0f,
        0.0f,  1.0f,  0.0f,
        0.0f,  1.0f,  0.0f,
        0.0f,  1.0f,  0.0f,
        0.0f,  1.0f,  0.0f,
        0.0f,  1.0f,  0.0f,

        // Bottom face normals
        0.0f, -1.0f,  0.0f,
        0.0f, -1.0f,  0.0f,
        0.0f, -1.0f,  0.0f,
        0.0f, -1.0f,  0.0f,
        0.0f, -1.0f,  0.0f,
        0.0f, -1.0f,  0.0f
    };
    
    return {GL_TRIANGLES, {{3, positions}, {3, normals}}};
}

Box::Box(glm::vec3 min, glm::vec3 max, glm::vec4 color)
    : Object{
          AssetManager::make<Shader>("box_shader", cfg::shaders_dir / "box.vert", cfg::shaders_dir / "box.frag"),
          AssetManager::make<Geometry>("box_geometry", box_geometry((max - min) / 2.f)),
      }, bb{std::make_unique<BoundingBox>(min, max)} {
    shader->set_uniform("color", color);

    glm::vec3 center = (min + max) / 2.f;
    set_model(glm::translate(center));
}

void Box::render() const {
    glCullFace(GL_FRONT);
    Object::render();
    glCullFace(GL_BACK);
}

MovingBox::MovingBox(glm::vec3 min, glm::vec3 max, glm::vec4 color) : Box(min, max, color), initial_bb{min, max} {}

constexpr float FREQUENCY = 0.5f;
constexpr float AMPLITUDE = 0.1f;

void MovingBox::update(float delta) {
    time += delta;

    float offset = std::sin(time * FREQUENCY) * AMPLITUDE;
    bb->min.x = initial_bb.min.x + offset;
    bb->max.x = initial_bb.max.x + offset;
    glm::vec3 center = (bb->min + bb->max) / 2.f;
    set_model(glm::translate(center));
}
