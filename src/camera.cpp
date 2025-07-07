#include "camera.h"

#include <glm/gtc/matrix_transform.hpp>


Camera::Camera(glm::vec3 position, float yaw, float pitch, int width, int height)
    : position{position}, yaw{yaw}, pitch{pitch} {
    update_vectors();
    set_projection(width, height);
}

void Camera::set_view() {
    view = glm::lookAt(position, position + front, up);
}

void Camera::set_projection(int width, int height) {
    projection = glm::perspective(FOV, static_cast<float>(width) / static_cast<float>(height), NEAR, FAR);
}

void Camera::update_vectors() {
    front = glm::normalize(glm::vec3{
        std::cos(yaw) * std::cos(pitch),
        std::sin(pitch),
        std::sin(yaw) * std::cos(pitch)
    });
    right = glm::normalize(glm::cross(front, WORLD_UP));
    up = glm::normalize(glm::cross(right, front));

    set_view();
}

void Camera::on_mouse_move(glm::vec2 offset) {
    offset *= SENSITIVITY;

    yaw   += offset.x;
    pitch += offset.y;

    // make sure that when pitch is out of bounds, screen doesn't get flipped
    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    // update Front, Right and Up Vectors using the updated Euler angles
    update_vectors();
}

void Camera::on_key_move(Camera::move m, float delta) {
    float x = delta * SPEED;
    switch (m) {
    case move::FORWARD:
        position += front * x;
        break;
    case move::BACKWARD:
        position -= front * x;
        break;
    case move::LEFT:
        position -= right * x;
        break;
    case move::RIGHT:
        position += right * x;
        break;
    }
    set_view();
}
