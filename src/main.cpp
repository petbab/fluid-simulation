#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "glfw.h"
#include "window.h"
#include "shader.h"
#include "config.h"
#include "geometry.h"

int main() {
    // Initialize GLFW
    GLFW glfw{};

    // Create window
    Window window{800, 600, "Fluid Simulation"};

    // Initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        throw std::runtime_error{"Failed to initialize GLAD"};

    Shader shader{cfg::shaders_dir/"shader.vert", cfg::shaders_dir/"shader.frag"};

    Geometry triangle = procedural::triangle();

    // Render loop
    while (!window.should_close()) {
        // Input
        window.process_input();

        // Clear screen
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw triangle
        shader.use();
        triangle.draw();

        // Swap buffers and poll events
        window.swap_buffers();
        glfwPollEvents();
    }

    return 0;
}