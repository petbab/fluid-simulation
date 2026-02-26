#include "object.h"


Object::Object(Shader *shader, Geometry *geometry) : shader{shader}, geometry{geometry} {}

void Object::render() const {
    assert(shader != nullptr);
    assert(geometry != nullptr);

    material.bind_ubo(UBO<void>::MATERIAL_UBO_BINDING);
    model.bind_ubo(UBO<void>::MODEL_UBO_BINDING);

    shader->use();
    geometry->draw();
}
