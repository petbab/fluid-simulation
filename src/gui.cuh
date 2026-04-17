#pragma once

#include "glfw.h"


class GUI {
public:
    explicit GUI(GLFWwindow* window);
    ~GUI();

    void update(float delta);
    void render();
};
