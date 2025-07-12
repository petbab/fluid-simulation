#include <glad/glad.h>
#include "application.h"
#include "config.h"
#include "shader.h"
#include "geometry.h"
#include "debug.h"


Application::Application(GLFWwindow *window, int width, int height)
    : window{window},
      camera{{0, 0, 10}, glm::radians(270.f), 0, width, height} {
    configure_window();
}

void Application::configure_window() {
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, on_resize);
    glfwSetCursorPosCallback(window, on_mouse_move);
    glfwSetKeyCallback(window, on_key_pressed);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void Application::run() {
    Shader ball_shader{cfg::shaders_dir/"instanced_ball.vert", cfg::shaders_dir/"instanced_ball.frag"};
    ball_shader.set_uniform("projection_frag", camera.get_projection());

    Shader axes_shader{cfg::shaders_dir/"axes.vert", cfg::shaders_dir/"axes.frag"};

    std::vector<float> positions{
        0., 0., 0.,
        2., 0., 0.,
        0., 2., 0.,
        0., 0., 2.,
        0., 2., 2.,
        2., 2., 0.,
        2., 0., 2.,
        2., 2., 2.,
    };
    InstancedGeometry ball_instances{procedural::quad(1, false, false), 1, {{3, positions}}};

    Geometry axes = procedural::axes(10);

    glEnable(GL_DEPTH_TEST);

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        const double current_time = glfwGetTime() * 1000.0; // from seconds to milliseconds
        const double delta = current_time - last_glfw_time;
        last_glfw_time = current_time;

        // Poll for and process events.
        glfwPollEvents();

        process_keyboard_input(static_cast<float>(delta));

        // Clear screen
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glCheckError();

        ball_shader.set_camera_uniforms(camera.get_view(), camera.get_projection());
        ball_shader.set_uniform("camera_pos", camera.get_position());
        ball_instances.draw();

        axes_shader.set_camera_uniforms(camera.get_view(), camera.get_projection());
        axes.draw();

        glfwSwapBuffers(window);
    }
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
