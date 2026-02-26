#include <glad/glad.h>
#include "fountain.h"
#include <glm/ext/matrix_transform.hpp>
#include <config.h>
#include <render/asset_manager.h>
#include <render/fluid.h>
#include <render/box.h>


struct fountain_force {
    __device__ float4 operator()(float4 pos) const {
        return (pos.x*pos.x + pos.z*pos.z < .2f && pos.y < 0.f)
            ? make_float4(0., 25., 0., 0.) : make_float4(0.f);
    };
};
static_assert(ExternalForce<fountain_force>);

void FountainApp::setup_scene() {
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

    glm::mat4 dragon_model{1.f};
    dragon_model = glm::translate(dragon_model, glm::vec3{1., -0.45, 0.});
    dragon_model = glm::rotate(dragon_model, -glm::pi<float>() * 0.5f, glm::vec3{1., 0., 0.});
    dragon_obj->set_model(dragon_model);

    /*
     * Fluid Box
     */
    Box *fluid_box = AssetManager::make<Box>(
        "fluid_box", glm::vec3{2.5, 0.7, 0.7}, glm::vec3{0.9});

    /*
     * Fluid
     */
    FluidSimulator::opts_t fluid_opts{{-1.5, 0., 0.}, 30, fluid_box->bounding_box(), {fluid_box, dragon_obj}};
    AssetManager::make<Fluid<CUDASPHSimulator<fountain_force>>>("fluid", fluid_opts);

    /*
     * Lights
     */
    lights = std::make_unique<LightArray>();
    lights->set_ambient_light(glm::vec3{0.1f});
    lights->add_directional_light(glm::vec3{0.2, 1., 0.4}, glm::vec3{0.2f}, glm::vec3{0.9f, 0.85, 0.8}, glm::vec3{1.0f});
    lights->add_point_light(glm::vec3{-0.7, 0.2, 0.4}, glm::vec3{0., 0., 0.1},
        glm::vec3{0.1f, 0.1, 0.8}, glm::vec3{1.0f}, 1., 0.7, 1.8);
    lights->upload_data();
}

void FountainApp::update_objects(float delta) {
    for (auto object : AssetManager::container<Object>())
        object->update(delta);
}
