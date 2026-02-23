#include "box.h"
#include <glm/gtx/transform.hpp>
#include "asset_manager.h"
#include "../config.h"

Box::Box(glm::vec3 half_size, glm::vec3 color)
    : Object{
          AssetManager::make<Shader>("box_shader", cfg::shaders_dir / "box.vert", cfg::shaders_dir / "lit.frag"),
          nullptr,
      }, bb{std::make_unique<BoundingBox>(-half_size, half_size)} {
    set_material(color, color, color, 32.f, 1.f);

    static unsigned box_counter = 0;
    geometry = AssetManager::make<Geometry>("box_geometry" + std::to_string(box_counter++),
        procedural::cube(half_size));
}

void Box::render() const {
    glCullFace(GL_FRONT);
    Object::render();
    glCullFace(GL_BACK);
}

void Box::set_model(const glm::mat4& m) {
    Object::set_model(m);
    bb->model = m;
    bb->model_inv = glm::inverse(m);
}
