#pragma once

#include <GLFW/glfw3.h>
#include "camera.h"


class Application {
public:
    Application(int width, int height, const char *title);

    void run();

private:
    static void on_resize(GLFWwindow* window, int width, int height);
    static void on_mouse_move(GLFWwindow* window, double x, double y);
    static void on_key_pressed(GLFWwindow* window, int key, int scancode, int action, int mods);

    GLFWwindow *window;
    Camera camera;
    double last_glfw_time = 0.;
};
