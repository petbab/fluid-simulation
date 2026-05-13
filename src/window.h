#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdexcept>
#include "debug.h"


/**
 * @brief RAII wrapper for a GLFW window and OpenGL context.
 *
 * Creates a GLFW window, initializes GLAD, and optionally sets up
 * a debug context in DEBUG builds.
 */
class Window {
public:
    /**
     * @brief Constructs and initializes the window.
     * @param width Initial width in pixels.
     * @param height Initial height in pixels.
     * @param title Window title.
     * @throws std::runtime_error if window or GLAD creation fails.
     */
    Window(int width, int height, const char *title) {
        window = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!window)
            throw std::runtime_error{"Failed to create GLFW window"};
        glfwMakeContextCurrent(window);

        // Initialize GLAD
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
            throw std::runtime_error{"Failed to initialize GLAD"};

#ifdef DEBUG
        int flags; glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
        if (flags & GL_CONTEXT_FLAG_DEBUG_BIT) {
            glEnable(GL_DEBUG_OUTPUT);
            glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
            glDebugMessageCallback(glDebugOutput, nullptr);
            glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
            glCheckError();
        }
#endif
    }

    /** @return Current window width in pixels. */
    int width() const {
        int w;
        glfwGetWindowSize(window, &w, nullptr);
        return w;
    }

    /** @return Current window height in pixels. */
    int height() const {
        int h;
        glfwGetWindowSize(window, nullptr, &h);
        return h;
    }

    /** @return Raw GLFW window handle. */
    GLFWwindow* get() { return window; }

private:
    GLFWwindow *window;  ///< GLFW window handle.
};
