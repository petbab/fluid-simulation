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

    Geometry(const Geometry&) = delete;
    Geometry& operator=(const Geometry&) = delete;
    Geometry(Geometry&&) noexcept;
    Geometry& operator=(Geometry&&) noexcept;
    ~Geometry();

    void draw() const;

private:
    GLenum mode;
    unsigned VBO = 0,
             VAO = 0;
    int vertices_count;

    friend class InstancedGeometry;
};

class InstancedGeometry {
public:
    InstancedGeometry(GLenum mode,
        const std::vector<VertexAttribute> &attributes,
        const std::vector<VertexAttribute> &instance_attributes);
    InstancedGeometry(Geometry geometry, std::size_t attribute_count,
        const std::vector<VertexAttribute> &instance_attributes);

    ~InstancedGeometry();

    void draw() const;

private:
    Geometry geometry;
    unsigned instanceVBO = 0;
    int instance_count;
};

namespace procedural {

Geometry triangle(bool color);

Geometry quad(float side_length, bool tex_coord, bool color);

Geometry axes(float half_size = 5.);

}
