#include "fluid.h"
#include "asset_manager.h"
#include "config.h"

Fluid::Fluid() : positions{
        0., 0., 0.,
        2., 0., 0.,
        0., 2., 0.,
        0., 0., 2.,
        0., 2., 2.,
        2., 2., 0.,
        2., 0., 2.,
        2., 2., 2.,
} {
    shader = AssetManager::make<Shader>(
        "instanced_ball_shader",
        cfg::shaders_dir/"instanced_ball.vert",
        cfg::shaders_dir/"instanced_ball.frag");
    geometry = AssetManager::make<InstancedGeometry>(
        "ball_geometry",
        procedural::quad(1, false, false), 1,
        std::vector{VertexAttribute{3, positions}});
}

void Fluid::update(double delta) {
    for (float &p : positions)
        p += static_cast<float>(delta) * 0.001f;
    dynamic_cast<const InstancedGeometry*>(geometry)->update_instance_data(positions);
}
