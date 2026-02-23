#include <glad/glad.h>
#include "tilting_box.h"

#include <glm/gtx/transform.hpp>
#include <render/asset_manager.h>
#include <render/box.h>
#include <render/fluid.h>


void TiltingBoxApp::setup_scene() {
    /*
     * Fluid Box
     */
    Box *fluid_box = AssetManager::make<Box>(
        "fluid_box", glm::vec3{2.3, 0.7, 0.6}, glm::vec3{0.9});

    /*
     * Fluid
     */
    FluidSimulator::opts_t fluid_opts{{0., 0., 0.}, 30, fluid_box->bounding_box(), {}};
    AssetManager::make<Fluid<FluidSim>>("fluid", fluid_opts);

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

void TiltingBoxApp::update_objects(float delta) {
    update_time += delta;

    Box *fluid_box = AssetManager::get<Box>("fluid_box");
    if (fluid_box != nullptr) {
        glm::mat4 fluid_box_model{1.f};
        fluid_box_model = glm::rotate(fluid_box_model, std::sin(update_time / 5.f) / 5.f, glm::vec3(0.f, 0.f, 1.f));
        fluid_box->set_model(fluid_box_model);
    }

    for (auto object : AssetManager::container<Object>())
        object->update(delta);
}
