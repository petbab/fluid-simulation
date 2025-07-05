#include "camera.h"

#include <glm/gtc/matrix_transform.hpp>


Camera::Camera(glm::vec3 position, glm::vec3 front, glm::vec2 window_size)
    : position{position}, front{front} {
    set_view();
    set_projection(window_size);
}

void Camera::set_view() {
    view = glm::lookAt(position, position + front, {0, 1, 0});
}

void Camera::set_projection(glm::vec2 window_size) {
    projection = glm::perspective(FOV, window_size.x / window_size.y, NEAR, FAR);
}
