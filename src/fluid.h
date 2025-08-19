#pragma once

#include <type_traits>
#include "object.h"
#include "fluid_simulator.h"
#include "asset_manager.h"
#include "config.h"


template<class S>
concept Simulator = std::is_base_of_v<FluidSimulator, S>;

template<Simulator S>
class Fluid : public Object {
public:
    Fluid(unsigned grid_count, BoundingBox bounding_box) : simulator{std::make_unique<S>(grid_count, bounding_box)} {
        shader = AssetManager::make<Shader>(
            "instanced_ball_shader",
            cfg::shaders_dir/"instanced_ball.vert",
            cfg::shaders_dir/"instanced_ball.frag");
        geometry = AssetManager::make<InstancedGeometry>(
            "ball_geometry",
            procedural::quad(1, false, false), 1,
            std::vector{VertexAttribute{3, simulator->get_position_data()}});
    }

    void update(double delta) override {
        simulator->update(delta);
        dynamic_cast<const InstancedGeometry*>(geometry)->update_instance_data(simulator->get_position_data());
    }

private:
    std::unique_ptr<FluidSimulator> simulator;
};
