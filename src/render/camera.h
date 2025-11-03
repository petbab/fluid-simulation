#pragma once

#include <glm/glm.hpp>
#include "ubo.h"


class Camera {
    static constexpr float FOV         = glm::radians(45.f);
    static constexpr float NEAR        = .1;
    static constexpr float FAR         = 100;
    static constexpr float SPEED       = 2.f;
    static constexpr float SENSITIVITY = 0.0005f;

    static constexpr glm::vec3 WORLD_UP{0, 1, 0};

public:
    Camera(glm::vec3 position, float yaw, float pitch, int width, int height);

    void update_window_size(int width, int height) { set_projection(width, height); }

    const glm::mat4& get_projection() const { return data.projection; }
    const glm::mat4& get_view() const { return data.view; }
    glm::vec3 get_position() const { return data.position; }

    void on_mouse_move(glm::vec2 offset);

    enum class move {
        FORWARD, BACKWARD, LEFT, RIGHT, UP, DOWN
    };
    void on_key_move(move m, float delta);

private:
    void set_view();
    void set_projection(int width, int height);
    void update_vectors();

    /*
layout(std140, binding = 0) uniform CameraData {
    mat4 projection;
    mat4 view;
    vec3 position;
} camera;
    */
    struct CameraData {
        glm::mat4 projection;
        glm::mat4 view;
        glm::vec3 position;
        float _pad = 0.f;
    };
    UBO<CameraData> ubo;
    CameraData data;
    glm::vec3 front, right, up;
    float yaw, pitch;
};
