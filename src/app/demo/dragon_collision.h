#pragma once
#include <app/application.h>


class DragonCollisionApp : public Application {
    using Application::Application;

protected:
    void setup_scene() override;
    void update_objects(float delta) override;
};
