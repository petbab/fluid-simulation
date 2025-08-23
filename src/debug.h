#pragma once

#include <glm/glm.hpp>
#include <glad/glad.h>
#include <iostream>
#include <stdexcept>
#include <vector>


std::ostream& operator<<(std::ostream &out, glm::vec3 v);

/**
 * @param delta milliseconds
 */
void print_fps(double delta);

void print_stats(const std::vector<float> &v, const std::string &name);

GLenum gl_check_error(const char *file, int line);
#ifdef DEBUG
#define glCheckError() gl_check_error(__FILE__, __LINE__)
#else
#define glCheckError()
#endif

void APIENTRY glDebugOutput(GLenum source, GLenum type, unsigned int id,
                            GLenum severity, GLsizei length, const char *message,
                            const void *userParam);
