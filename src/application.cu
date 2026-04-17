#include <glad/glad.h>
#include "application.h"
#include <config.h>
#include <debug.h>
#include <imgui.h>
#include <render/asset_manager.h>
#include <render/fluid.h>


Application::Application(GLFWwindow *window, int width, int height)
    : window{window},
      camera{{0, 0, 2.5}, glm::radians(270.f), 0, width, height} {
    configure_window();
    gui = std::make_unique<GUI>(window);
}

void Application::init() {
    setup_scene();
}

void Application::configure_window() {
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, on_resize);
    glfwSetCursorPosCallback(window, on_mouse_move);
    glfwSetKeyCallback(window, on_key_pressed);
    glfwSetMouseButtonCallback(window, on_mouse_button);
}

void Application::run() {
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        const double current_time = glfwGetTime();
        const float delta = current_time - last_glfw_time;
        last_glfw_time = current_time;

        // Poll for and process events.
        glfwPollEvents();

        gui->update(delta);

        update(delta);

        // Clear screen
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glCheckError();

        render_scene();

        gui->render();

        glfwSwapBuffers(window);
    }
}

void Application::render_scene() {
    camera.bind_ubo(UBO<void>::CAMERA_UBO_BINDING);
    lights->bind_ubo(UBO<void>::LIGHTS_UBO_BINDING);

    for (auto object : AssetManager::container<Object>())
        object->render();
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

void Application::on_resize(GLFWwindow *window, int width, int height) {
    glViewport(0, 0, width, height);
    glCheckError();

    Application* app = app_from_window(window);
    app->camera.update_window_size(width, height);
}

void Application::on_mouse_move(GLFWwindow *window, double x, double y) {
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

void Application::on_key_pressed(GLFWwindow *window, int key, int, int action, int) {
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
