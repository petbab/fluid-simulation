#pragma once

#include <filesystem>
#include <glad/glad.h>
#include <glm/glm.hpp>


class Shader {
    static constexpr std::string VIEW_UNIFORM = "view";
    static constexpr std::string PROJECTION_UNIFORM = "projection";

public:
    Shader(const char *vertex, const char *fragment);
    Shader(const std::filesystem::path& vertex, const std::filesystem::path& fragment);

    ~Shader();

    void use() const;

    GLint get_uniform_location(const std::string &name) const;

    void set_camera_uniforms(const glm::mat4 &view, const glm::mat4 &projection) const;

private:
    unsigned program;
};
