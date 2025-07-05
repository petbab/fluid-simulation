#pragma once

#include <glm/glm.hpp>
#include <string>


class Camera {
    static constexpr float FOV = glm::radians(45.f);
    static constexpr float NEAR = .1;
    static constexpr float FAR = 100;

public:
    Camera(glm::vec3 position, glm::vec3 front, glm::vec2 window_size);

    void update_window_size(glm::vec2 window_size) { set_projection(window_size); }

    const glm::mat4& get_projection() const { return projection; }
    const glm::mat4& get_view() const { return view; }

private:
    void set_view();
    void set_projection(glm::vec2 window_size);

    glm::vec3 position, front;
    glm::mat4 projection, view;
};
