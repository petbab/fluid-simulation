#pragma once

#include <GLFW/glfw3.h>


class GUI {
public:
    explicit GUI(GLFWwindow* window);
    ~GUI();

    void loop_start();
    void loop_end();
};
