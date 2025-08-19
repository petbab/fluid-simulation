#include <glad/glad.h>
#include "application.h"
#include "config.h"
#include "debug.h"
#include "asset_manager.h"
#include "fluid.h"
#include "box.h"
#include "dfsph.h"


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
    Box *fluid_box = AssetManager::make<Box>("fluid_box", glm::vec3{-0.5, -0.5, -0.5}, glm::vec3{0.5, 0.5, 0.5},
                                             glm::vec4{0.45, 0.4, 0.4, 1.});
    objects.push_back(fluid_box);

    objects.push_back(AssetManager::make<Fluid<DFSPHSimulator>>("fluid", 10, *fluid_box));

    auto *axes_shader = AssetManager::make<Shader>(
        "axes_shader",
        cfg::shaders_dir/"axes.vert",
        cfg::shaders_dir/"axes.frag");
    auto *axes_geom = AssetManager::make<Geometry>("axes_geometry", procedural::axes(10));
    objects.push_back(
        AssetManager::make<Object>("axes", axes_shader, axes_geom)
    );
}

void Application::render_scene() {
    for (auto &object : objects)
        object->render();
}

void Application::update(double delta) {
    process_keyboard_input(static_cast<float>(delta));
    for (auto &object : objects)
        object->update(delta);
}

void Application::on_resize(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
    glCheckError();

    auto application = static_cast<Application*>(glfwGetWindowUserPointer(window));
    application->camera.update_window_size(width, height);
}

void Application::on_mouse_move(GLFWwindow *window, double x, double y) {
    auto application = static_cast<Application*>(glfwGetWindowUserPointer(window));

    glm::vec2 pos{x, y};
    if (application->first_mouse_move)
    {
        application->last_mouse_pos = pos;
        application->first_mouse_move = false;
        return;
    }

    glm::vec2 offset = pos - application->last_mouse_pos;
    offset.y = -offset.y; // reversed since y-coordinates go from bottom to top
    application->camera.on_mouse_move(offset);

    application->last_mouse_pos = pos;
}

void Application::on_key_pressed(GLFWwindow *window, int key, int, int, int) {
    if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window, GL_TRUE);
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
    else if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        camera.on_key_move(Camera::move::UP, delta);
    else if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
        camera.on_key_move(Camera::move::DOWN, delta);
}
