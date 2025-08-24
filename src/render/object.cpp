#include "object.h"


Object::Object(Shader *shader, Geometry *geometry) : shader{shader}, geometry{geometry} {}

void Object::render() const {
    assert(shader != nullptr);
    assert(geometry != nullptr);

    shader->use();
    geometry->draw();
}

void Object::update(double) {}

void Object::set_model(const glm::mat4 &m) {
    assert(shader != nullptr);
    shader->set_uniform("model", m);
}
