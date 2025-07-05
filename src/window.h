#pragma once

#include <GLFW/glfw3.h>
#include <stdexcept>
#include <glm/glm.hpp>


class Window {
public:
    Window(int width, int height, const char *title, GLFWframebuffersizefun resize_fun) {
        window = glfwCreateWindow(width, height, title, nullptr, nullptr);
        if (!window)
            throw std::runtime_error{"Failed to create GLFW window"};

        glfwMakeContextCurrent(window);

        if (resize_fun != nullptr)
            glfwSetFramebufferSizeCallback(window, resize_fun);
    }

    void set_window_user_pointer(void *ptr) { glfwSetWindowUserPointer(window, ptr); }

    bool should_close() const { return glfwWindowShouldClose(window); }

    void swap_buffers() { glfwSwapBuffers(window); }

    void process_input() {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
    }

    glm::vec2 size() const {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        return {static_cast<float>(width), static_cast<float>(height)};
    }

private:
    GLFWwindow *window;
};
