#include <glad/glad.h>
#include "application.h"
#include "config.h"
#include "shader.h"
#include "geometry.h"


Application::Application(int width, int height, const char *title)
    : camera{{0, 0, 10}, glm::radians(270.f), 0, width, height} {
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
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    Shader shader{cfg::shaders_dir/"shader.vert", cfg::shaders_dir/"disk.frag"};
    Shader axes_shader{cfg::shaders_dir/"axes.vert", cfg::shaders_dir/"axes.frag"};

    Geometry object = procedural::quad(5);
    Geometry axes = procedural::axes(20);

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

        shader.set_camera_uniforms(camera.get_view(), camera.get_projection());
        object.draw();

        axes_shader.set_camera_uniforms(camera.get_view(), camera.get_projection());
        axes.draw();

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

void Application::on_key_pressed(GLFWwindow *window, int key, int scancode, int action, int mods) {
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
}
