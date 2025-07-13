#include "object.h"


Object::Object(const Shader &shader, const Geometry &geometry) : shader{shader}, geometry{geometry} {}

void Object::render() const {
    shader.use();
    geometry.draw();
}

void Object::update(double) {}
