#include "shader.h"
#include "debug.h"

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
        throw std::runtime_error{"Shader compilation failed: " + std::string{info_log} };
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

Shader::Shader(const std::filesystem::path &vertex, const std::filesystem::path &fragment)
    : Shader{read_file(vertex).c_str(), read_file(fragment).c_str()} {}

Shader::~Shader() {
    glDeleteProgram(program);
    glCheckError();
}

void Shader::use() const {
    glUseProgram(program);
    glCheckError();
}

GLint Shader::get_uniform_location(const std::string &name) const {
    use();
    GLint l = glGetUniformLocation(program, name.c_str());
    glCheckError();
    return l;
}

void Shader::set_camera_uniforms(const glm::mat4 &view, const glm::mat4 &projection) const {
    GLint view_loc = get_uniform_location(VIEW_UNIFORM);
    GLint projection_loc = get_uniform_location(PROJECTION_UNIFORM);

    glUniformMatrix4fv(view_loc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projection_loc, 1, GL_FALSE, glm::value_ptr(projection));

    glCheckError();
}
