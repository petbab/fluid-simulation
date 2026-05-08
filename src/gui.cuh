#pragma once

#include "glfw.h"


class GUI {
public:
    explicit GUI(GLFWwindow* window, const std::string& name);
    ~GUI();

    void update(float delta);
    void render();

private:
    float vis_min = 0.f, vis_max = 0.f;
    std::string name;
};
