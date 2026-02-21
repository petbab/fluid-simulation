#include "box.h"
#include <glm/gtx/transform.hpp>
#include "asset_manager.h"
#include "../config.h"

Box::Box(glm::vec3 min, glm::vec3 max, glm::vec3 color)
    : Object{
          AssetManager::make<Shader>("box_shader", cfg::shaders_dir / "box.vert", cfg::shaders_dir / "lit.frag"),
          nullptr,
      }, bb{std::make_unique<BoundingBox>(min, max)} {
    glm::vec3 center = (min + max) / 2.f;
    set_model(glm::translate(center));
    set_material(color, color, color, 32.f, 1.f);

    static unsigned box_counter = 0;
    geometry = AssetManager::make<Geometry>("box_geometry" + std::to_string(box_counter++),
        procedural::cube((max - min) / 2.f));
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
