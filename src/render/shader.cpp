#include "shader.h"
#include "../debug.h"

#include <stdexcept>
#include <fstream>
#include <glm/gtc/type_ptr.hpp>


static unsigned compile_shader(unsigned int type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    // Check for compilation errors
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(shader, 512, nullptr, info_log);
        throw std::runtime_error{"Shader compilation failed: " + std::string{info_log} };
    }

    glCheckError();

    return shader;
}

Shader::Shader(const char *vertex, const char *fragment) {
    // Compile shaders
    unsigned vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex);
    unsigned fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment);

    // Create shader program
    program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    // Check for linking errors
    int success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(program, 512, nullptr, info_log);
        throw std::runtime_error{"Shader linkage failed: " + std::string{info_log} };
    }

    // Clean up shaders
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    glCheckError();
}

static std::string read_file(const std::filesystem::path &path) {
    std::ifstream file{path};
    if (!file.is_open())
        throw std::runtime_error{"Error opening file: " + path.string()};
    return {
        std::istreambuf_iterator<char>{file},
        std::istreambuf_iterator<char>{}
    };
}

Shader::Shader(
    const std::filesystem::path &vertex,
    const std::filesystem::path &fragment
) try : Shader{read_file(vertex).c_str(), read_file(fragment).c_str()} {
} catch (const std::exception& e) {
    throw std::runtime_error{
        "Creating shader from: " + vertex.filename().string() + ", "
        + fragment.filename().string() + ": " + std::string{e.what()}};
}

Shader::~Shader() {
    glDeleteProgram(program);
    glCheckError();
}

void Shader::use() const {
    glUseProgram(program);
    glCheckError();
}

GLint Shader::get_uniform_location(const std::string &name) {
    auto it = uniform_cache.find(name);
    if (it != uniform_cache.end())
        return it->second;

    use();
    GLint location = glGetUniformLocation(program, name.c_str());
    if (location == -1)
        throw std::runtime_error{"location of uniform '" + name + "' not found"};

    glCheckError();
    return uniform_cache[name] = location;
}

void Shader::set_uniform(const std::string& name, bool v) {
    GLint loc = get_uniform_location(name);
    use();
    glUniform1i(loc, v);
    glCheckError();
}

void Shader::set_uniform(const std::string& name, float v) {
    GLint loc = get_uniform_location(name);
    use();
    glUniform1f(loc, v);
    glCheckError();
}

void Shader::set_uniform(const std::string &name, glm::vec3 v) {
    GLint loc = get_uniform_location(name);
    use();
    glUniform3f(loc, v.x, v.y, v.z);
    glCheckError();
}

void Shader::set_uniform(const std::string &name, glm::vec4 v) {
    GLint loc = get_uniform_location(name);
    use();
    glUniform4f(loc, v.x, v.y, v.z, v.w);
    glCheckError();
}

void Shader::set_uniform(const std::string &name, const glm::mat4 &m) {
    GLint loc = get_uniform_location(name);
    use();
    glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(m));
    glCheckError();
}
