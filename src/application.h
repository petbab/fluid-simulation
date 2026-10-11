#pragma once

#include <memory>
#include <GLFW/glfw3.h>
#include <render/camera.h>
#include <render/light.h>
#include <cuda/SPH/sph.cuh>
#include "gui.cuh"
#include "cli.h"
#include "render/fluid.h"
#include "render/shader.h"


class Application {
public:
    using FluidSim = CUDASPHSimulator;
    static constexpr float DEFAULT_TIME_STEP = 0.01;

    Application(GLFWwindow *window, int width, int height, const std::string& name, const RunOptions& opts = {});
    virtual ~Application();

    void init();
    void run();

protected:
    void configure_window();
    void render_scene();
    void update(float delta);

    virtual void setup_scene() {}
    virtual void update_objects(float delta) {}

    void set_capture_mouse(bool capture_mouse);

    static void on_resize(GLFWwindow* window, int width, int height);
    static void on_mouse_move(GLFWwindow* window, double x, double y);
    static void on_key_pressed(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void on_mouse_button(GLFWwindow* window, int button, int action, int mods);

    void process_keyboard_input(float delta);

    void run_headless();
    bool stop_reached(int step, double sim_time, std::chrono::steady_clock::time_point t0) const;

    void init_fancy();
    void init_fancy_textures(int width, int height);
    void render_fancy() const;
    void render_texture() const;

    GLFWwindow *window;
    std::unique_ptr<GUI> gui;

    Camera camera;
    std::unique_ptr<LightArray> lights;

    double last_glfw_time = 0.;
    bool first_mouse_move = true;
    glm::vec2 last_mouse_pos{0.f};
    bool paused = true;
    bool captured_mouse = false;

    RunOptions opts;
    std::string app_name;

private:
    // Fancy rendering
    bool rendering_fancy = false;

    GLuint depth_tex = 0;
    GLuint fbo = 0;
    Shader *instanced_ball_unlit = nullptr;
    Fluid<FluidSim>* fluid = nullptr;

    enum class show_texture_t {
        depth,
        none,
    };
    static const std::vector<const char*> show_texture_strings;
    show_texture_t show_texture = show_texture_t::none;
    Shader *show_texture_shader = nullptr;
    GLuint empty_vao = 0;

    friend class GUI;
};
