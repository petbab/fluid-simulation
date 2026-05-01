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

    auto *fluid = AssetManager::get<Fluid<CUDASPHSimulator>>("fluid");
    assert(fluid != nullptr);
    CUDASPHSimulator &fluid_sim = fluid->get_simulator();

    ImGui::Text("Fluid Particles: %d", fluid_sim.fluid_particles);
    ImGui::Text("Boundary Particles: %d", fluid_sim.boundary_particles);
    ImGui::Text("FPS: %.2f", 1.f / delta);

    auto [searched, total] = fluid_sim.tuning_stats();
    ImGui::Text("Searched Configurations: %i/%i", searched, total);

    if (ImGui::SliderFloat("Tuning Budget", &fluid_sim.tuning_budget,
        0.01f, 10.0f, "%.2f", ImGuiSliderFlags_Logarithmic))
        fluid_sim.set_tuning_budget(fluid_sim.tuning_budget);

    ImGui::End();
    ImGui::EndFrame();
}

void GUI::render() {
    // Rendering
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
