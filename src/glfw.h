#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdexcept>


/**
 * @brief RAII wrapper for GLFW initialization and termination.
 *
 * Constructs glfwInit() and destroys glfwTerminate().
 * Configures an OpenGL 4.5 core profile context.
 */
class GLFW {
public:
    GLFW() {
        // Initialize GLFW
        if (!glfwInit())
            throw std::runtime_error{"Failed to initialize GLFW"};

        // Configure GLFW
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef DEBUG
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);
#endif
    }

    ~GLFW() {
        glfwTerminate();
    }
};
