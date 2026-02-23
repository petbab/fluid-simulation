#pragma once

#include "application.h"


class TiltingBoxApp : public Application {
    using Application::Application;

protected:
    void setup_scene() override;
    void update_objects(float delta) override;

private:
    float update_time = 0.f;
};
