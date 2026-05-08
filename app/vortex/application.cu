#include <glad/glad.h>
#include "application.h"
#include <glm/ext/matrix_transform.hpp>
#include <config.h>
#include <render/asset_manager.h>
#include <render/fluid.h>
#include <render/box.h>


static constexpr float WORLD_SCALE = 1.5f;

void VortexApp::setup_scene() {
    camera.set_position(camera.get_position() * WORLD_SCALE);

    auto *lit_shader = AssetManager::make<Shader>("lit_shader",
        cfg::shaders_dir/"lit.vert",
        cfg::shaders_dir/"lit.frag");

    /*
     * Dragon
     */
    auto *dragon_geom = AssetManager::make<Geometry>(
        "dragon_geometry", Geometry::from_file(cfg::models_dir / "stanford-dragon.obj"));
    auto *dragon_obj = AssetManager::make<Object>("dragon", lit_shader, dragon_geom);
    dragon_obj->set_material(glm::vec3{1., 0., 1.}, glm::vec3{1., 0., 1.}, glm::vec3{1., 0., 1.}, 32., 1.);

    glm::mat4 dragon_model = glm::scale(glm::mat4{1.f}, glm::vec3{WORLD_SCALE});
    dragon_model = glm::translate(dragon_model, glm::vec3{.75, -0.45, 0.});
    dragon_model = glm::rotate(dragon_model, -glm::pi<float>() * 0.5f, glm::vec3{0., 1., 0.});
    dragon_model = glm::rotate(dragon_model, -glm::pi<float>() * 0.5f, glm::vec3{1., 0., 0.});
    dragon_obj->set_model(dragon_model);

    /*
     * Fluid Box
     */
    Box *fluid_box = AssetManager::make<Box>(
        "fluid_box", glm::vec3{1., 0.7, 1.} * WORLD_SCALE, glm::vec3{0.9});

    /*
     * Fluid
     */
    FluidSimulator::opts_t fluid_opts{
        glm::vec3{-.25, 0., 0.} * WORLD_SCALE, 50,
        fluid_box->bounding_box(), {dragon_obj, fluid_box},
        "([](float4 pos) { return (pos.y < -0.5f) ? 2.f * normalize(make_float4(pos.z, 0.f, -pos.x, 0.f)) : make_float4(0.f); })"
    };
    AssetManager::make<Fluid<FluidSim>>("fluid", fluid_opts);

    /*
     * Lights
     */
    lights = std::make_unique<LightArray>();
    lights->set_ambient_light(glm::vec3{0.1f});
    lights->add_directional_light(glm::vec3{0.2, 1., 0.4}, glm::vec3{0.2f}, glm::vec3{0.9f, 0.85, 0.8}, glm::vec3{1.0f});
    lights->add_point_light(glm::vec3{-0.7, 0.2, 0.4} * WORLD_SCALE, glm::vec3{0., 0., 0.1},
        glm::vec3{0.1f, 0.1, 0.8}, glm::vec3{1.0f}, 1., 0.7, 1.8);
    lights->upload_data();
}

void VortexApp::update_objects(float delta) {
    for (auto object : AssetManager::container<Object>())
        object->update(delta);
}
