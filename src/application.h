#pragma once

#include <GLFW/glfw3.h>
#include <vector>
#include "render/camera.h"
#include "render/object.h"


class Application {
public:
    Application(GLFWwindow *window, int width, int height);

    void run();

private:
    void configure_window();
    void setup_scene();
    void render_scene();
    void update(float delta);
    void update_objects(float delta);

    static void on_resize(GLFWwindow* window, int width, int height);
    static void on_mouse_move(GLFWwindow* window, double x, double y);
    static void on_key_pressed(GLFWwindow* window, int key, int scancode, int action, int mods);

    void process_keyboard_input(float delta);

    GLFWwindow *window;
    Camera camera;

    std::vector<Object*> objects;

    double last_glfw_time = 0.;
    bool first_mouse_move = true;
    glm::vec2 last_mouse_pos{0.f};

#ifdef DEBUG
    bool paused = true;
#else
    bool paused = false;
#endif
};
