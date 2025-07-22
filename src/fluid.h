#pragma once

#include "object.h"


class Fluid : public Object {
public:
    Fluid();

    void update(double delta) override;

private:
    std::vector<float> positions;
};
