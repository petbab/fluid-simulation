#include <glad/glad.h>
#include "application.h"
#include "render/shader.h"
#include <config.h>
#include <debug.h>
#include <imgui.h>
#include <render/asset_manager.h>
#include <render/fluid.h>
#include <cuda/SPH/snapshot.cuh>
#include <cuda/tuning/config_loader.h>
#include <chrono>
#include <fstream>
#include <vector>


Application::Application(GLFWwindow *window, int width, int height, const std::string& name, const RunOptions& opts)
    : window{window},
      camera{{0, 0, 2.5}, glm::radians(270.f), 0, width, height},
      opts{opts},
      app_name{name} {
    configure_window();
    if (!opts.headless)
        gui = std::make_unique<GUI>(window, name);
}

Application::~Application() {
    glDeleteTextures(1, &depth_tex);
    glDeleteFramebuffers(1, &fbo);
    glDeleteVertexArrays(1, &empty_vao);
}

void Application::init() {
    setup_scene();

    fluid = AssetManager::get<Fluid<FluidSim>>("fluid");
    if (fluid == nullptr)
        return;
    CUDASPHSimulator& fluid_sim = fluid->get_simulator();

    if (opts.snapshot_load) {
        auto err = Snapshot::load(*opts.snapshot_load, app_name,
                                  fluid_sim.particle_data, fluid_sim.get_fluid_particles());
        if (!err.empty())
            throw std::runtime_error("Snapshot load failed: " + err);
    }

    if (opts.warmup_iters) {
        fluid_sim.set_tuning_budget(0.f);
        for (int i = 0; i < opts.warmup_iters; ++i)
            update_objects(opts.fixed_dt);
        fluid_sim.reset_tuning();
    }

    if (opts.frozen_config) {
        fluid_sim.set_frozen_config(
            load_config_json(*opts.frozen_config, *fluid_sim.step_tuner.get_tuner(),
                             fluid_sim.step_tuner.get_kernel()));
    }

    fluid_sim.step_tuner.set_searcher(opts.searcher);
    fluid_sim.set_tuning_budget(opts.tuning_budget);

    if (opts.ktt_output) {
        fluid_sim.set_result_out(opts.ktt_output);
    }

    init_fancy();
}

void Application::configure_window() {
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, on_resize);
    glfwSetCursorPosCallback(window, on_mouse_move);
    glfwSetKeyCallback(window, on_key_pressed);
    glfwSetMouseButtonCallback(window, on_mouse_button);
}

void Application::run() {
    if (opts.headless) {
        run_headless();
        return;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        const double current_time = glfwGetTime();
        const float delta = current_time - last_glfw_time;
        last_glfw_time = current_time;

        // Poll for and process events.
        glfwPollEvents();

        if (gui)
            gui->update(delta);

        update(delta);

        // Clear screen
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glCheckError();

        render_scene();

        if (gui)
            gui->render();

        glfwSwapBuffers(window);
    }
}

void Application::run_headless() {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    double sim_time = 0.0;
    int step = 0;

    std::ofstream log;
    if (opts.log_csv) {
        log.open(*opts.log_csv);
        log << "step,sim_time,wall_dt_ms,scheduled_tune";
        if (opts.log_metrics) log << ",mean_speed,ke";
        log << '\n';
    }

    auto* fluid = AssetManager::get<Fluid<FluidSim>>("fluid");
    CUDASPHSimulator* fluid_sim = fluid ? &fluid->get_simulator() : nullptr;

    while (!stop_reached(step, sim_time, t0)) {
        auto t_step = clock::now();
        update_objects(opts.fixed_dt);
        sim_time += opts.fixed_dt;
        auto wall_dt = std::chrono::duration<double, std::milli>(clock::now() - t_step).count();

        if (log && fluid_sim) {
            log << step << ',' << sim_time << ',' << wall_dt
                << ',' << (fluid_sim->was_scheduled_step() ? 1 : 0);
            if (opts.log_metrics) {
                auto [v, ke] = fluid_sim->compute_state_metrics();
                log << ',' << v << ',' << ke;
            }
            log << '\n';
        }
        ++step;
    }

    if (opts.snapshot_save && fluid_sim) {
        auto err = Snapshot::save(*opts.snapshot_save, app_name,
                                  fluid_sim->particle_data, fluid_sim->get_fluid_particles());
        if (!err.empty())
            throw std::runtime_error("Snapshot save failed: " + err);
    }
}

bool Application::stop_reached(int step, double sim_time, std::chrono::steady_clock::time_point t0) const {
    switch (opts.stop_kind) {
    case RunOptions::StopKind::Iters:
        return step >= static_cast<int>(opts.stop_value);
    case RunOptions::StopKind::SimTime:
        return sim_time >= opts.stop_value;
    case RunOptions::StopKind::WallTime: {
        auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return elapsed >= opts.stop_value;
    }
    default:
        return false;
    }
}

void Application::render_scene() {
    if (rendering_fancy) {
        render_fancy();
        return;
    }

    camera.bind_ubo(UBO<void>::CAMERA_UBO_BINDING);
    lights->bind_ubo(UBO<void>::LIGHTS_UBO_BINDING);

    for (auto object : AssetManager::container<Object>())
        object->render();
}

void Application::init_fancy() {
    instanced_ball_unlit = AssetManager::make<Shader>("instanced_ball_unlit_shader",
        cfg::shaders_dir / "instanced_ball.vert",
        cfg::shaders_dir / "instanced_ball_unlit.frag");

    show_texture_shader = AssetManager::make<Shader>("show_texture_shader",
        cfg::shaders_dir / "show_texture.vert",
        cfg::shaders_dir / "show_texture.frag");

    glCreateFramebuffers(1, &fbo); glCheckError();

    glCreateVertexArrays(1, &empty_vao);

    int w, h;
    glfwGetWindowSize(window, &w, &h);
    init_fancy_textures(w, h);
}

void Application::init_fancy_textures(int width, int height) {
    glDeleteTextures(1, &depth_tex);

    glCreateTextures(GL_TEXTURE_2D, 1, &depth_tex); glCheckError();
    glTextureStorage2D(depth_tex, 1, GL_DEPTH_COMPONENT32F, width, height); glCheckError();
    glTextureParameteri(depth_tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST); glCheckError();
    glTextureParameteri(depth_tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST); glCheckError();
    glTextureParameteri(depth_tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glCheckError();
    glTextureParameteri(depth_tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); glCheckError();
    glTextureParameteri(depth_tex, GL_TEXTURE_COMPARE_MODE, GL_NONE); glCheckError();

    glNamedFramebufferTexture(fbo, GL_DEPTH_ATTACHMENT, depth_tex, 0); glCheckError();
}

void Application::render_fancy() const {
    camera.bind_ubo(UBO<void>::CAMERA_UBO_BINDING);
    lights->bind_ubo(UBO<void>::LIGHTS_UBO_BINDING);

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
    glClearColor(0.f, 0.f, 0.f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    for (auto object : AssetManager::container<Object>())
        if (object != fluid)
            object->render();
    fluid->render();

    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    if (show_texture != show_texture_t::none)
        render_texture();
}

void Application::render_texture() const {
    show_texture_shader->use();

    switch (show_texture) {
    case show_texture_t::depth:
        show_texture_shader->set_uniform("color", false);
        show_texture_shader->set_uniform("depth", true);
        glBindTextureUnit(0, depth_tex);
    case show_texture_t::none:
        break;
    }
    glCheckError();

    glBindVertexArray(empty_vao); glCheckError();
    glDrawArrays(GL_TRIANGLES, 0, 3); glCheckError();
    glBindVertexArray(0);
}

void Application::update(float delta) {
    process_keyboard_input(delta);
    if (!paused)
        update_objects(delta);
}

static Application* app_from_window(GLFWwindow *window) {
    return static_cast<Application*>(glfwGetWindowUserPointer(window));
}

void Application::set_capture_mouse(bool capture_mouse) {
    if (capture_mouse != captured_mouse) {
        captured_mouse = capture_mouse;
        glfwSetInputMode(window, GLFW_CURSOR, capture_mouse ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
    }
}

void Application::on_resize(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    glCheckError();

    Application* app = app_from_window(window);
    app->camera.update_window_size(width, height);
    app->init_fancy_textures(width, height);
}

void Application::on_mouse_move(GLFWwindow* window, double x, double y) {
    if (ImGui::GetIO().WantCaptureMouse)
        return;

    Application* app = app_from_window(window);

    if (!app->captured_mouse) {
        app->first_mouse_move = true;
        return;
    }

    glm::vec2 pos{x, y};
    if (app->first_mouse_move)
    {
        app->last_mouse_pos = pos;
        app->first_mouse_move = false;
        return;
    }

    glm::vec2 offset = pos - app->last_mouse_pos;
    offset.y = -offset.y; // reversed since y-coordinates go from bottom to top
    app->camera.on_mouse_move(offset);

    app->last_mouse_pos = pos;
}

void Application::on_key_pressed(GLFWwindow* window, int key, int, int action, int) {
    if (action != GLFW_PRESS)
        return;

    Application *app = app_from_window(window);

    if (key == GLFW_KEY_ESCAPE) {
        if (!app->captured_mouse) {
            glfwSetWindowShouldClose(window, GL_TRUE);
            return;
        }

        app->set_capture_mouse(false);
    }

    if (ImGui::GetIO().WantCaptureKeyboard)
        return;

    auto *fluid = AssetManager::get<Fluid<FluidSim>>("fluid");
    switch (key) {
    case GLFW_KEY_R:
        if (fluid != nullptr)
            fluid->reset();
        break;
    case GLFW_KEY_SPACE:
        app->paused = !app->paused;
        break;
    case GLFW_KEY_RIGHT:
        if (app->paused)
            app->update_objects(DEFAULT_TIME_STEP);
        break;
    case GLFW_KEY_B:
        if (fluid != nullptr)
            fluid->toggle_show_boundary();
        break;
    }
}

void Application::on_mouse_button(GLFWwindow* window, int button, int action, int mods) {
    if (ImGui::GetIO().WantCaptureMouse)
        return;

    if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT) {
        Application *app = app_from_window(window);
        app->set_capture_mouse(true);
    }
}

void Application::process_keyboard_input(float delta) {
    if (ImGui::GetIO().WantCaptureKeyboard)
        return;

    // Camera movement
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.on_key_move(Camera::move::FORWARD, delta);
    else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.on_key_move(Camera::move::BACKWARD, delta);
    else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.on_key_move(Camera::move::LEFT, delta);
    else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.on_key_move(Camera::move::RIGHT, delta);
    else if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        camera.on_key_move(Camera::move::UP, delta);
    else if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        camera.on_key_move(Camera::move::DOWN, delta);
}

const auto Application::show_texture_strings = std::vector{
    "Depth",
    "None",
};
