#pragma once
#include <app/application.h>


class FountainApp : public Application {
    using Application::Application;

protected:
    void setup_scene() override;
    void update_objects(float delta) override;
};
