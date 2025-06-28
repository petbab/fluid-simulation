#include "shader.h"

#include <stdexcept>
#include <filesystem>
#include <fstream>
#include "glad/glad.h"


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
}

void Shader::use() const {
    glUseProgram(program);
}
