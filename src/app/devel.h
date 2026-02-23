#pragma once
#include "application.h"


class DevelApp : public Application {
    using Application::Application;

protected:
    void setup_scene() override;
    void update_objects(float delta) override;
};
