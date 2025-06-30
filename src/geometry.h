#pragma once

#include <vector>
#include <glad/glad.h>


class Geometry {
public:
    Geometry(GLenum mode, const std::vector<float> &positions, const std::vector<float> &colors);

    ~Geometry();

    void draw() const;

private:
    GLenum mode;
    std::vector<float> vertex_data;
    unsigned VBO = -1,
             VAO = -1;
    int vertices_count;
};

namespace procedural {

Geometry triangle();

}
