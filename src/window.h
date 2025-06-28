#pragma once

#include <GLFW/glfw3.h>
#include <stdexcept>


void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

class Window {
public:
    Window(int width, int height, const char *title) {
        window = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!window)
            throw std::runtime_error{"Failed to create GLFW window"};

        glfwMakeContextCurrent(window);
        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    }

    bool should_close() const { return glfwWindowShouldClose(window); }

    void swap_buffers() { glfwSwapBuffers(window); }

    void process_input() {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
    }

private:
    GLFWwindow *window;
};
