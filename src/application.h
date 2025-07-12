#pragma once

#include <GLFW/glfw3.h>
#include <vector>
#include <memory>
#include "camera.h"
#include "object.h"


class Application {
public:
    Application(GLFWwindow *window, int width, int height);

    void run();

private:
    void configure_window();
    void setup_scene();
    void render_scene();
    void update(double delta);

    static void on_resize(GLFWwindow* window, int width, int height);
    static void on_mouse_move(GLFWwindow* window, double x, double y);
    static void on_key_pressed(GLFWwindow* window, int key, int scancode, int action, int mods);

    void process_keyboard_input(float delta);

    template<class T, class... Args>
    static auto* make_asset(auto &asset_list, Args&&... args) {
        asset_list.push_back(std::make_unique<T>(std::forward<Args>(args)...));
        return asset_list.rbegin()->get();
    }

    template<class T = Shader, class... Args>
    Shader* make_shader(Args&&... args) {
        return make_asset<T>(shader_list, std::forward<Args>(args)...);
    }
    template<class T = Geometry, class... Args>
    Geometry* make_geometry(Args&&... args) {
        return make_asset<T>(geometry_list, std::forward<Args>(args)...);
    }
    template<class T = Object, class... Args>
    Object* make_object(Args&&... args) {
        return make_asset<T>(object_list, std::forward<Args>(args)...);
    }

    GLFWwindow *window;
    Camera camera;
    std::vector<std::unique_ptr<Shader>> shader_list;
    std::vector<std::unique_ptr<Geometry>> geometry_list;
    std::vector<std::unique_ptr<Object>> object_list;

    double last_glfw_time = 0.;
    bool first_mouse_move = true;
    glm::vec2 last_mouse_pos{0.f};
};
