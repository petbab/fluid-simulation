#include <glad/glad.h>
#include "application.h"
#include "config.h"
#include "shader.h"
#include "geometry.h"


Application::Application(int width, int height, const char *title)
    : camera{{0, 0, 5}, {0, 0, -1}, width, height} {
    window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window)
        throw std::runtime_error{"Failed to create GLFW window"};
    glfwMakeContextCurrent(window);

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        throw std::runtime_error{"Failed to initialize GLAD"};
}

void Application::run() {
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, on_resize);
    glfwSetCursorPosCallback(window, on_mouse_move);
    glfwSetKeyCallback(window, on_key_pressed);

    Shader shader{cfg::shaders_dir/"shader.vert", cfg::shaders_dir/"disk.frag"};

    Geometry object = procedural::quad(.1f);

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        const double current_time = glfwGetTime() * 1000.0; // from seconds to milliseconds
        const double delta = current_time - last_glfw_time;
        last_glfw_time = current_time;

        // Poll for and process events.
        glfwPollEvents();

        // TODO: process input

        // Clear screen
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        shader.set_camera_uniforms(camera.get_view(), camera.get_projection());
        object.draw();

        glfwSwapBuffers(window);
    }
}

void Application::on_resize(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
    auto application = static_cast<Application*>(glfwGetWindowUserPointer(window));
    application->camera.update_window_size(width, height);
}

void Application::on_mouse_move(GLFWwindow *window, double x, double y) {
    auto application = static_cast<Application*>(glfwGetWindowUserPointer(window));
}

void Application::on_key_pressed(GLFWwindow *window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE)
        glfwSetWindowShouldClose(window, GL_TRUE);
}
