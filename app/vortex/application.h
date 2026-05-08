#pragma once
#include <application.h>


class VortexApp : public Application {
    using Application::Application;

protected:
    void setup_scene() override;
    void update_objects(float delta) override;

private:
    glm::vec3 last_camera_pos;
};
