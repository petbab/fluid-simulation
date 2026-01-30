#pragma once

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cuda_runtime_api.h>


std::ostream& operator<<(std::ostream &out, glm::vec3 v);

/**
 * @param delta milliseconds
 */
void print_fps(float delta);

void print_stats(const std::vector<float> &v, const std::string &name);
void print_stats(const std::vector<glm::vec3> &v, const std::string &name);

GLenum gl_check_error(const char *file, int line);
#ifdef DEBUG
#define glCheckError() gl_check_error(__FILE__, __LINE__)
#else
#define glCheckError()
#endif

void APIENTRY glDebugOutput(GLenum source, GLenum type, unsigned int id,
                            GLenum severity, GLsizei length, const char *message,
                            const void *userParam);

inline void cuda_check_error(const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA error " << cudaGetErrorName(err) << ": "
            << cudaGetErrorString(err) << " at " << file << " (" << line << ")" << std::endl;
}
#ifdef DEBUG
#define cudaCheckError() cuda_check_error(__FILE__, __LINE__)
#else
#define cudaCheckError()
#endif
