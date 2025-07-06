#pragma once

#include <glm/glm.hpp>
#include <string>


class Camera {
    static constexpr float FOV = glm::radians(45.f);
    static constexpr float NEAR = .1;
    static constexpr float FAR = 100;

public:
    Camera(glm::vec3 position, glm::vec3 front, int width, int height);

    void update_window_size(int width, int height) { set_projection(width, height); }

    const glm::mat4& get_projection() const { return projection; }
    const glm::mat4& get_view() const { return view; }

private:
    void set_view();
    void set_projection(int width, int height);

    glm::vec3 position, front;
    glm::mat4 projection, view;
};
