#include "fluid.h"
#include "asset_manager.h"
#include "config.h"


Fluid::Fluid(unsigned grid_count, BoundingBox bounding_box)
    : simulation{grid_count, bounding_box} {

    shader = AssetManager::make<Shader>(
        "instanced_ball_shader",
        cfg::shaders_dir/"instanced_ball.vert",
        cfg::shaders_dir/"instanced_ball.frag");
    geometry = AssetManager::make<InstancedGeometry>(
        "ball_geometry",
        procedural::quad(1, false, false), 1,
        std::vector{VertexAttribute{3, simulation.get_position_data()}});
}

void Fluid::update(double delta) {
    simulation.update(delta);
    dynamic_cast<const InstancedGeometry*>(geometry)->update_instance_data(simulation.get_position_data());
}
