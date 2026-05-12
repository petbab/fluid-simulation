#pragma once

#include <glm/glm.hpp>
#include "ubo.h"


/**
 * @brief First-person style camera with UBO-backed data.
 *
 * Manages view and projection matrices, position, and orientation (yaw/pitch).
 * Camera data is uploaded to a GPU uniform buffer object for use in shaders.
 */
class Camera {
    static constexpr float FOV         = glm::radians(45.f);  ///< Field of view in radians.
    static constexpr float NEAR        = .1;                  ///< Near clipping plane.
    static constexpr float FAR         = 100;                 ///< Far clipping plane.
    static constexpr float SPEED       = 2.f;                 ///< Movement speed multiplier.
    static constexpr float SENSITIVITY = 0.0005f;             ///< Mouse sensitivity multiplier.

    static constexpr glm::vec3 WORLD_UP{0, 1, 0};  ///< World-space up direction.

public:
    /**
     * @brief Constructs a camera.
     * @param position Initial world-space position.
     * @param yaw Initial yaw angle in radians.
     * @param pitch Initial pitch angle in radians.
     * @param width Viewport width in pixels.
     * @param height Viewport height in pixels.
     */
    Camera(glm::vec3 position, float yaw, float pitch, int width, int height);

    /**
     * @brief Updates the projection matrix after a window resize.
     * @param width New viewport width.
     * @param height New viewport height.
     */
    void update_window_size(int width, int height) { set_projection(width, height); }

    /** @return The projection matrix. */
    const glm::mat4& get_projection() const { return data.projection; }
    /** @return The view matrix. */
    const glm::mat4& get_view() const { return data.view; }
    /** @return The camera position in world space. */
    glm::vec3 get_position() const { return data.position; }

    /**
     * @brief Sets the camera position and updates the view matrix.
     * @param position New world-space position.
     */
    void set_position(glm::vec3 position);

    /**
     * @brief Updates orientation based on mouse movement.
     * @param offset Mouse delta (x, y).
     */
    void on_mouse_move(glm::vec2 offset);

    /** @brief Movement directions for keyboard input. */
    enum class move {
        FORWARD, BACKWARD, LEFT, RIGHT, UP, DOWN
    };

    /**
     * @brief Moves the camera based on keyboard input.
     * @param m Movement direction.
     * @param delta Time delta in seconds.
     */
    void on_key_move(move m, float delta);

    /**
     * @brief Binds the camera UBO to the given binding point.
     * @param binding UBO binding index.
     */
    void bind_ubo(unsigned binding) const { ubo.bind(binding); }

private:
    void set_view();
    void set_projection(int width, int height);
    void update_vectors();

    /*
layout(std140, binding = 0) uniform CameraData {
    mat4 projection;
    mat4 view;
    vec3 eye_position;
};
    */
    /** @brief Data layout matching the CameraData UBO in shaders. */
    struct CameraData {
        glm::mat4 projection;  ///< Projection matrix.
        glm::mat4 view;        ///< View matrix.
        glm::vec3 position;    ///< Camera position.
        float _pad = 0.f;      ///< Padding for std140 alignment.
    };
    UBO<CameraData> ubo;  ///< Uniform buffer object for camera data.
    CameraData data;      ///< CPU-side camera data.
    glm::vec3 front, right, up;  ///< Camera basis vectors.
    float yaw, pitch;            ///< Euler angles in radians.
};
