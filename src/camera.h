#pragma once

#include <glm/glm.hpp>
#include <string>


class Camera {
    static constexpr float FOV         = glm::radians(45.f);
    static constexpr float NEAR        = .1;
    static constexpr float FAR         = 100;
    static constexpr float SPEED       = 0.05f;
    static constexpr float SENSITIVITY = 0.001f;

    static constexpr glm::vec3 WORLD_UP{0, 1, 0};

public:
    Camera(glm::vec3 position, float yaw, float pitch, int width, int height);

    void update_window_size(int width, int height) { set_projection(width, height); }

    const glm::mat4& get_projection() const { return projection; }
    const glm::mat4& get_view() const { return view; }

    void on_mouse_move(glm::vec2 offset);

    enum class move {
        FORWARD, BACKWARD, LEFT, RIGHT
    };
    void on_key_move(move m, float delta);

private:
    void set_view();
    void set_projection(int width, int height);

    void update_vectors();

    glm::vec3 position, front, right, up;
    glm::mat4 projection, view;
    float yaw, pitch;
};
