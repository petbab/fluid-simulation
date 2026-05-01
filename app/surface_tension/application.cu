#include <glad/glad.h>
#include "application.h"

#include <render/asset_manager.h>
#include <render/box.h>
#include <render/fluid.h>


void SurfaceTensionApp::setup_scene() {
    /*
     * Fluid Box
     */
    Box *fluid_box = AssetManager::make<Box>(
        "fluid_box", glm::vec3{0.8, 0.8, 0.8}, glm::vec3{0.9});

    /*
     * Fluid
     */
    FluidSimulator::opts_t fluid_opts{
        {0., 0., 0.}, 30, fluid_box->bounding_box(), {},
        "([](float4 pos) { return make_float4(0., 9.81f, 0., 0.); })"
    };
    AssetManager::make<Fluid<FluidSim>>("fluid", fluid_opts);

    /*
     * Lights
     */
    lights = std::make_unique<LightArray>();
    lights->set_ambient_light(glm::vec3{0.1f});
    lights->add_directional_light(glm::vec3{0.2, 1., 0.4}, glm::vec3{0.2f}, glm::vec3{0.9f, 0.85, 0.8}, glm::vec3{1.0f});
    lights->add_point_light(glm::vec3{-0.6, 0.2, 0.4}, glm::vec3{0., 0., 0.1},
        glm::vec3{0.1f, 0.1, 0.8}, glm::vec3{1.0f}, 1., 0.7, 1.8);
    lights->upload_data();
}

void SurfaceTensionApp::update_objects(float delta) {
    for (auto object : AssetManager::container<Object>())
        object->update(delta);
}
