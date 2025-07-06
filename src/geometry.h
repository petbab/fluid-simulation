#pragma once

#include <vector>
#include <glad/glad.h>


struct VertexAttribute {
    unsigned elem_size;
    const std::vector<float> &data;
};

class Geometry {
public:
    Geometry(GLenum mode, const std::vector<VertexAttribute> &attributes);

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

Geometry triangle(bool color = true);

Geometry quad(float side_length, bool color = true);

Geometry axes(float half_size = 5.);

}
