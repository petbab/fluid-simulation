#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "glfw.h"
#include "window.h"
#include "shader.h"
#include "config.h"
#include "geometry.h"
#include "camera.h"


void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    auto camera = static_cast<Camera*>(glfwGetWindowUserPointer(window));
    camera->update_window_size({width, height});
}

int main() {
    // Initialize GLFW
    GLFW glfw{};

    // Create window
    Window window{800, 600, "Fluid Simulation", framebuffer_size_callback};

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        throw std::runtime_error{"Failed to initialize GLAD"};

    Shader shader{cfg::shaders_dir/"shader.vert", cfg::shaders_dir/"disk.frag"};

    Camera camera{{0, 0, 5}, {0, 0, -1}, window.size()};
    window.set_window_user_pointer(static_cast<void*>(&camera));
    shader.set_camera_uniforms(camera.get_view(), camera.get_projection());

    Geometry object = procedural::quad(.1f);

    // Render loop
    while (!window.should_close()) {
        // Input
        window.process_input();

        // Clear screen
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw triangle
        shader.use();
        object.draw();

        // Swap buffers and poll events
        window.swap_buffers();
        glfwPollEvents();
    }

    return 0;
}