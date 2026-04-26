#pragma once

#include <type_traits>
#include "object.h"
#include <simulation/fluid_simulator.h>
#include "asset_manager.h"
#include <config.h>
#include <cuda/SPH/sph.cuh>


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
        geometry_a = AssetManager::make<InstancedGeometry>(
            "ball_geometry_a",
            procedural::quad(1, false, false), 1,
            std::vector{VertexAttribute{4, simulator->get_position_data()}});
        geometry_b = AssetManager::make<InstancedGeometry>(
            "ball_geometry_b",
            procedural::quad(1, false, false), 1,
            std::vector{VertexAttribute{4, simulator->get_position_data()}});
        geometry = geometry_a;

        if constexpr (std::is_base_of_v<CUDASPHSimulator, S>)
            dynamic_cast<CUDASPHSimulator*>(simulator.get())->init_positions(
                geometry_a->get_instance_vbo(), geometry_b->get_instance_vbo());
    }

    void update(float delta) override {
        simulator->update(delta);

        // Don't update if using CUDA (works directly with the data on the GPU, no transfer needed)
        // if constexpr (!std::is_base_of_v<CUDASPHSimulator, S>)
        //     inst_geom()->update_instance_data(simulator->get_position_data());

        std::swap(geometry_a, geometry_b);
        geometry = geometry_a;
    }

    void render() const override {
        shader->use();
        shader->set_uniform("fluid_particles", simulator->get_fluid_particles());
        shader->set_uniform("show_boundary", show_boundary);
        simulator->visualize(shader);
        Object::render();
    }

    void reset() {
        // TODO: check swap logic
        simulator->reset();
        inst_geom()->update_instance_data(simulator->get_position_data());
    }

    void toggle_show_boundary() { show_boundary = !show_boundary; }

private:
    const InstancedGeometry* inst_geom() const { return dynamic_cast<const InstancedGeometry*>(geometry); }

    const S& get_simulator() const { return dynamic_cast<const S&>(*simulator); }
    S& get_simulator() { return dynamic_cast<S&>(*simulator); }

    InstancedGeometry *geometry_a, *geometry_b;
    std::unique_ptr<FluidSimulator> simulator;
    bool show_boundary = false;

    friend class GUI;
};
