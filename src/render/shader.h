#pragma once

#include <filesystem>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>


class Shader {
public:
    Shader(const char *vertex, const char *fragment);
    Shader(const std::filesystem::path& vertex, const std::filesystem::path& fragment);

    ~Shader();

    void use() const;

    void set_uniform(const std::string &name, bool v);
    void set_uniform(const std::string &name, unsigned n);
    void set_uniform(const std::string &name, float v);
    void set_uniform(const std::string &name, glm::vec3 v);
    void set_uniform(const std::string &name, glm::vec4 v);
    void set_uniform(const std::string &name, const glm::mat4 &m);

private:
    GLint get_uniform_location(const std::string &name);

    unsigned program;
    std::unordered_map<std::string, GLint> uniform_cache;
};
