#pragma once

#include "ubo.h"


/**
 * @brief Model transformation matrix stored in a UBO.
 *
 * Stores both the model matrix and its inverse-transpose (for normal transformation).
 */
class Model {
    /** @brief Data layout matching the Model UBO in shaders. */
    struct ModelData {
        glm::mat4 model;      ///< Model-to-world transformation matrix.
        glm::mat3x4 model_it; ///< Inverse-transpose of the upper 3x3 of the model matrix.

        /**
         * @brief Constructs ModelData from a model matrix.
         * @param m The model matrix.
         */
        ModelData(const glm::mat4& m) : model{m},
            model_it{glm::mat3x4(transpose(inverse(glm::mat3{m})))} {}
    };

public:
    /** @brief Constructs an identity model transformation. */
    Model() : Model(glm::mat4{1.0f}) {}

    /**
     * @brief Constructs a model transformation.
     * @param m Initial model matrix.
     */
    explicit Model(const glm::mat4 &m) : data{m} {
        ubo.upload_data(&data);
    }

    /**
     * @brief Updates the model matrix and uploads to GPU.
     * @param m New model matrix.
     */
    void set(const glm::mat4 &m) {
        data = m;
        ubo.upload_data(&data);
    }

    /**
     * @brief Binds the model UBO to the given binding point.
     * @param binding UBO binding index.
     */
    void bind_ubo(unsigned binding) const { ubo.bind(binding); }

    /** @return The current model matrix. */
    const glm::mat4& get_model() const { return data.model; }

private:
    UBO<ModelData> ubo;  ///< Uniform buffer object for model data.
    ModelData data;      ///< CPU-side model data.
};
