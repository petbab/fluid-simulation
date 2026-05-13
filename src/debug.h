#pragma once

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cuda_runtime_api.h>


/**
 * @brief Stream output operator for glm::vec4.
 * @param out Output stream.
 * @param v Vector to print.
 * @return Reference to the output stream.
 */
std::ostream& operator<<(std::ostream &out, glm::vec4 v);

/**
 * @brief Prints frames-per-second based on frame delta.
 * @param delta Frame duration in milliseconds.
 */
void print_fps(float delta);

/**
 * @brief Prints statistics (min, max, mean) of a float vector.
 * @param v Vector of values.
 * @param name Label printed with the statistics.
 */
void print_stats(const std::vector<float> &v, const std::string &name);

/**
 * @brief Prints statistics (min, max, mean) of a vec4 vector.
 * @param v Vector of vec4 values.
 * @param name Label printed with the statistics.
 */
void print_stats(const std::vector<glm::vec4> &v, const std::string &name);

/**
 * @brief Checks and prints the current OpenGL error.
 * @param file Source file where the check is performed.
 * @param line Source line where the check is performed.
 * @return The OpenGL error enum.
 */
GLenum gl_check_error(const char *file, int line);

#ifdef DEBUG
/** @brief Macro that calls gl_check_error with the current file and line. */
#define glCheckError() gl_check_error(__FILE__, __LINE__)
#else
/** @brief No-op macro when DEBUG is not defined. */
#define glCheckError()
#endif

/**
 * @brief OpenGL debug message callback.
 *
 * Registered with glDebugMessageCallback in debug builds.
 */
void APIENTRY glDebugOutput(GLenum source, GLenum type, unsigned int id,
                            GLenum severity, GLsizei length, const char *message,
                            const void *userParam);

/**
 * @brief Checks and prints the last CUDA error.
 * @param file Source file where the check is performed.
 * @param line Source line where the check is performed.
 */
inline void cuda_check_error(const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA error " << cudaGetErrorName(err) << ": "
            << cudaGetErrorString(err) << " at " << file << " (" << line << ")" << std::endl;
}

#ifdef DEBUG
/** @brief Macro that calls cuda_check_error with the current file and line. */
#define cudaCheckError() cuda_check_error(__FILE__, __LINE__)
#else
/** @brief No-op macro when DEBUG is not defined. */
#define cudaCheckError()
#endif
