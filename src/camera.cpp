#include "camera.h"

#include <glm/gtc/matrix_transform.hpp>


Camera::Camera(glm::vec3 position, glm::vec3 front, int width, int height)
    : position{position}, front{front} {
    set_view();
    set_projection(width, height);
}

void Camera::set_view() {
    view = glm::lookAt(position, position + front, {0, 1, 0});
}

void Camera::set_projection(int width, int height) {
    projection = glm::perspective(FOV, static_cast<float>(width) / static_cast<float>(height), NEAR, FAR);
}
