#pragma once

#include "ubo.h"


class Model {
    struct ModelData {
        glm::mat4 model;
        glm::mat3x4 model_it;

        ModelData(const glm::mat4& m) : model{m},
            model_it{glm::mat3x4(transpose(inverse(glm::mat3{m})))} {}
    };

public:
    Model() : Model(glm::mat4{1.0f}) {}
    explicit Model(const glm::mat4 &m) : data{m} {
        ubo.upload_data(&data);
    }

    void set(const glm::mat4 &m) {
        data = m;
        ubo.upload_data(&data);
    }

    void bind_ubo(unsigned binding) const { ubo.bind(binding); }

    const glm::mat4& get_model() const { return data.model; }

private:
    UBO<ModelData> ubo;
    ModelData data;
};
