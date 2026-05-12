#pragma once

#include <filesystem>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>


/**
 * @brief OpenGL shader program wrapper.
 *
 * Compiles vertex and fragment shaders, links them into a program,
 * and provides methods for setting uniforms with automatic caching.
 */
class Shader {
public:
    /**
     * @brief Constructs a shader from source strings.
     * @param vertex Vertex shader source code.
     * @param fragment Fragment shader source code.
     */
    Shader(const char *vertex, const char *fragment);

    /**
     * @brief Constructs a shader from file paths.
     * @param vertex Path to the vertex shader file.
     * @param fragment Path to the fragment shader file.
     */
    Shader(const std::filesystem::path& vertex, const std::filesystem::path& fragment);

    ~Shader();

    /** @brief Activates this shader program for rendering. */
    void use() const;

    /**
     * @brief Sets a boolean uniform.
     * @param name Uniform name.
     * @param v Boolean value.
     */
    void set_uniform(const std::string &name, bool v);

    /**
     * @brief Sets an unsigned integer uniform.
     * @param name Uniform name.
     * @param n Unsigned integer value.
     */
    void set_uniform(const std::string &name, unsigned n);

    /**
     * @brief Sets a float uniform.
     * @param name Uniform name.
     * @param v Float value.
     */
    void set_uniform(const std::string &name, float v);

    /**
     * @brief Sets a vec3 uniform.
     * @param name Uniform name.
     * @param v vec3 value.
     */
    void set_uniform(const std::string &name, glm::vec3 v);

    /**
     * @brief Sets a vec4 uniform.
     * @param name Uniform name.
     * @param v vec4 value.
     */
    void set_uniform(const std::string &name, glm::vec4 v);

    /**
     * @brief Sets a mat4 uniform.
     * @param name Uniform name.
     * @param m mat4 value.
     */
    void set_uniform(const std::string &name, const glm::mat4 &m);

private:
    /**
     * @brief Retrieves (and caches) the location of a uniform.
     * @param name Uniform name.
     * @return OpenGL uniform location.
     * @throws std::runtime_error if the uniform is not found.
     */
    GLint get_uniform_location(const std::string &name);

    unsigned program;  ///< OpenGL shader program identifier.
    std::unordered_map<std::string, GLint> uniform_cache;  ///< Cache of uniform locations.
};
