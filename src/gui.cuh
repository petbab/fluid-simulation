#pragma once

#include "glfw.h"


/**
 * @brief ImGui overlay for the application.
 *
 * Provides a debug/visualization panel showing simulation statistics
 * and controls for the fluid simulator.
 */
class GUI {
public:
    /**
     * @brief Constructs the GUI.
     * @param window GLFW window to render into.
     * @param name Window title for the GUI panel.
     */
    explicit GUI(GLFWwindow* window, const std::string& name);
    ~GUI();

    /**
     * @brief Updates GUI state (called once per frame).
     * @param delta Time step in seconds.
     */
    void update(float delta);

    /** @brief Renders the GUI panel. */
    void render();

private:
    float vis_min = 0.f, vis_max = 0.f;  ///< Current visualization min/max values.
    std::string name;                    ///< Panel title.
};
