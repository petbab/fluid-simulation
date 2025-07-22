#include "object.h"


Object::Object(const Shader *shader, const Geometry *geometry) : shader{shader}, geometry{geometry} {}

void Object::render() const {
    assert(shader != nullptr);
    assert(geometry != nullptr);

    shader->use();
    geometry->draw();
}

void Object::update(double) {}
