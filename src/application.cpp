#include <glad/glad.h>
#include "application.h"
#include "config.h"
#include "debug.h"
#include "render/asset_manager.h"
#include "render/fluid.h"
#include "render/box.h"
#include "simulation/SPH/sph.h"


using FluidSim = SPHSimulator;

static constexpr float DEFAULT_TIME_STEP = 0.01;

Application::Application(GLFWwindow *window, int width, int height)
    : window{window},
      camera{{0, 0, 2.5}, glm::radians(270.f), 0, width, height} {
    configure_window();
    setup_scene();
}

void Application::configure_window() {
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, on_resize);
    glfwSetCursorPosCallback(window, on_mouse_move);
    glfwSetKeyCallback(window, on_key_pressed);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void Application::run() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        const double current_time = glfwGetTime();
        const double delta = current_time - last_glfw_time;
        last_glfw_time = current_time;

        // Poll for and process events.
        glfwPollEvents();
        update(delta);

        // Clear screen
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glCheckError();

        render_scene();
        glfwSwapBuffers(window);
    }
}

void Application::setup_scene() {
    Box *fluid_box = AssetManager::make<Box>("fluid_box", glm::vec3{-1, -0.75, -0.75}, glm::vec3{1, 0.75, 0.75},
                                             glm::vec4{0.65, 0.6, 0.6, 1.});
    objects.push_back(fluid_box);

    objects.push_back(AssetManager::make<Fluid<FluidSim>>("fluid", 20, fluid_box->bounding_box()));

//    auto *axes_shader = AssetManager::make<Shader>(
//        "axes_shader",
//        cfg::shaders_dir/"axes.vert",
//        cfg::shaders_dir/"axes.frag");
//    auto *axes_geom = AssetManager::make<Geometry>("axes_geometry", procedural::axes(10));
//    objects.push_back(
//        AssetManager::make<Object>("axes", axes_shader, axes_geom)
//    );
}

void Application::render_scene() {
    for (auto &object : objects)
        object->render();
}

void Application::update(double delta) {
    process_keyboard_input(static_cast<float>(delta));
    if (!paused)
        update_objects(delta);
}

static Application* app_from_window(GLFWwindow *window) {
    return static_cast<Application*>(glfwGetWindowUserPointer(window));
}

void Application::on_resize(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
    glCheckError();

    Application* app = app_from_window(window);
    app->camera.update_window_size(width, height);
}

void Application::on_mouse_move(GLFWwindow *window, double x, double y) {
    Application* app = app_from_window(window);

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

void Application::on_key_pressed(GLFWwindow *window, int key, int, int action, int) {
    if (action != GLFW_PRESS)
        return;

    Application *app = app_from_window(window);
    switch (key) {
    case GLFW_KEY_ESCAPE:
        glfwSetWindowShouldClose(window, GL_TRUE);
        break;
    case GLFW_KEY_R:
        AssetManager::get<Fluid<SPHSimulator>>("fluid")->reset();
        break;
    case GLFW_KEY_SPACE:
        app->paused = !app->paused;
        break;
    case GLFW_KEY_RIGHT:
        if (app->paused)
            app->update_objects(DEFAULT_TIME_STEP);
        break;
    }
}

void Application::process_keyboard_input(float delta) {
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

void Application::update_objects(double delta) {
    for (auto object : objects)
        object->update(delta);
}
