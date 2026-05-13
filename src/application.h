#pragma once

#include <memory>
#include <GLFW/glfw3.h>
#include <render/camera.h>
#include <render/light.h>
#include <cuda/SPH/sph.cuh>
#include "gui.cuh"


/**
 * @brief Main application class managing the simulation loop, rendering, and input.
 *
 * Owns the GLFW window, GUI, camera, lights, and fluid simulator.
 * Derived classes override setup_scene() and update_objects() to define
 * application-specific behavior.
 */
class Application {
public:
    using FluidSim = CUDASPHSimulator;          ///< Alias for the fluid simulator type.
    static constexpr float DEFAULT_TIME_STEP = 0.01;  ///< Default simulation time step.

    /**
     * @brief Constructs the application.
     * @param window GLFW window handle.
     * @param width Initial window width.
     * @param height Initial window height.
     * @param name Application name (used by the GUI).
     */
    Application(GLFWwindow *window, int width, int height, const std::string& name);
    virtual ~Application() = default;

    /** @brief Initializes the application (scene, camera, GUI). */
    void init();

    /** @brief Runs the main loop until the window closes. */
    void run();

protected:
    /** @brief Configures GLFW callbacks and window settings. */
    void configure_window();

    /** @brief Renders the current scene. */
    void render_scene();

    /**
     * @brief Updates simulation and objects.
     * @param delta Time step in seconds.
     */
    void update(float delta);

    /** @brief Override to set up scene objects, lights, and simulator. */
    virtual void setup_scene() {}

    /**
     * @brief Override to update custom objects.
     * @param delta Time step in seconds.
     */
    virtual void update_objects(float delta) {}

    /**
     * @brief Enables or disables mouse capture.
     * @param capture_mouse If true, captures the cursor.
     */
    void set_capture_mouse(bool capture_mouse);

    /** @brief GLFW framebuffer resize callback. */
    static void on_resize(GLFWwindow* window, int width, int height);
    /** @brief GLFW cursor position callback. */
    static void on_mouse_move(GLFWwindow* window, double x, double y);
    /** @brief GLFW keyboard callback. */
    static void on_key_pressed(GLFWwindow* window, int key, int scancode, int action, int mods);
    /** @brief GLFW mouse button callback. */
    static void on_mouse_button(GLFWwindow* window, int button, int action, int mods);

    /**
     * @brief Processes keyboard input for camera movement.
     * @param delta Time step in seconds.
     */
    void process_keyboard_input(float delta);

    GLFWwindow *window;                         ///< GLFW window handle.
    std::unique_ptr<GUI> gui;                   ///< ImGui overlay.

    Camera camera;                              ///< Scene camera.
    std::unique_ptr<LightArray> lights;         ///< Scene lights.

    double last_glfw_time = 0.;                 ///< Time of last frame.
    bool first_mouse_move = true;               ///< True until first mouse event.
    glm::vec2 last_mouse_pos{0.f};              ///< Last mouse position.
    bool paused = true;                         ///< Simulation paused state.
    bool captured_mouse = false;                ///< Mouse capture state.
};
