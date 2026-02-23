#pragma once

#include <type_traits>
#include "object.h"
#include <simulation/fluid_simulator.h>
#include "asset_manager.h"
#include <config.h>
#include <cuda/simulator.h>


template<class S>
concept Simulator = std::is_base_of_v<FluidSimulator, S>;

template<Simulator S>
class Fluid : public Object {
public:
    Fluid(const FluidSimulator::opts_t &opts) : simulator{std::make_unique<S>(opts)} {
        shader = AssetManager::make<Shader>(
            "instanced_ball_shader",
            cfg::shaders_dir/"instanced_ball.vert",
            cfg::shaders_dir/"instanced_ball.frag");
        geometry = AssetManager::make<InstancedGeometry>(
            "ball_geometry",
            procedural::quad(1, false, false), 1,
            std::vector{VertexAttribute{3, simulator->get_position_data()}});

        if constexpr (std::is_base_of_v<CUDASimulator, S>)
            dynamic_cast<CUDASimulator*>(simulator.get())->init_buffer(inst_geom()->get_instance_vbo());
    }
    Fluid(unsigned grid_count, const BoundingBox &bounding_box) : Fluid({grid_count, grid_count, grid_count},bounding_box) {}

    void update(float delta) override {
        simulator->update(delta);
        update_geometry();
    }

    void render() const override {
        shader->use();
        shader->set_uniform("fluid_particles", simulator->get_fluid_particles());
        shader->set_uniform("show_boundary", show_boundary);
        simulator->visualize(shader);
        Object::render();
    }

    void reset() {
        simulator->reset();
        update_geometry();
    }

    void toggle_show_boundary() { show_boundary = !show_boundary; }

private:
    void update_geometry() {
        if constexpr (!std::is_base_of_v<CUDASimulator, S>)
            inst_geom()->update_instance_data(simulator->get_position_data());
    }

    const InstancedGeometry* inst_geom() const { return dynamic_cast<const InstancedGeometry*>(geometry); }

    std::unique_ptr<FluidSimulator> simulator;
    bool show_boundary = false;
};
