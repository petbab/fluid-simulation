#include "gui.cuh"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "cuda/SPH/sph.cuh"
#include "render/asset_manager.h"
#include "render/fluid.h"


GUI::GUI(GLFWwindow* window) {
    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    // Setup Platform/Renderer backends
    // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init();
}

GUI::~GUI() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void GUI::update(float delta) {
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();

    ImGui::NewFrame();
    ImGui::Begin("Fluid Simulation");

    ///////////////////////////////////
    //            Tuning             //
    ///////////////////////////////////
    ImGui::SeparatorText("Tuning");

    ImGui::Text("FPS: %f", 1.f / delta);

    auto *fluid = AssetManager::get<Fluid<CUDASPHSimulator>>("fluid");
    assert(fluid != nullptr);
    CUDASPHSimulator &fluid_sim = fluid->get_simulator();

    auto [searched, total] = fluid_sim.tuning_stats();
    ImGui::Text("Searched Configurations: %i/%i", searched, total);

    if (ImGui::SliderFloat("Tuning Budget", &fluid_sim.tuning_budget,
        0.01f, 10.0f, "%.2f", ImGuiSliderFlags_Logarithmic))
        fluid_sim.set_tuning_budget(fluid_sim.tuning_budget);

    if (ImGui::Button("Reset Tuning"))
        fluid_sim.reset_tuning();

    ///////////////////////////////////
    //          Simulation           //
    ///////////////////////////////////
    ImGui::SeparatorText("Simulation");

    ImGui::Text("Fluid Particles: %d", fluid_sim.fluid_particles);
    ImGui::Text("Boundary Particles: %d", fluid_sim.boundary_particles);

    ImGui::Checkbox("Show Boundary [B]", &fluid->show_boundary);

    auto &visualizer = fluid_sim.particle_data_visualizer;
    int viz_mode = static_cast<int>(visualizer.mode);
    if (ImGui::Combo("Visualization", &viz_mode,
        ParticleDataVisualizer::mode_strings.data(),
        static_cast<int>(ParticleDataVisualizer::mode_strings.size()))) {
        visualizer.set_mode(static_cast<ParticleDataVisualizer::mode_t>(viz_mode));
        vis_min = visualizer.min;
        vis_max = visualizer.max;
    }

    if (visualizer.normalize) {
        float range[2] = {visualizer.min, visualizer.max};
        if (ImGui::SliderFloat2("Range", range, vis_min, vis_max, "%.3f")) {
            visualizer.min = range[0];
            visualizer.max = range[1];
        }
    }

    if (ImGui::Button("Reset Fluid"))
        fluid->reset();

    ImGui::End();
    ImGui::EndFrame();
}

void GUI::render() {
    // Rendering
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
