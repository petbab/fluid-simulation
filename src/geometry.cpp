#include <cassert>
#include "geometry.h"
#include "debug.h"


Geometry::Geometry(GLenum mode, const std::vector<VertexAttribute> &attributes)
    : mode{mode}, vertices_count{static_cast<int>(attributes[0].data.size() / attributes[0].elem_size)} {
    assert(!attributes.empty());

    int stride = 0;
    for (const auto &attr : attributes)
        stride += static_cast<int>(attr.elem_size);

    std::vector<float> vertex_data;
    vertex_data.reserve(stride * vertices_count);
    for (int i = 0; i < vertices_count; ++i)
        for (const auto &attr : attributes)
            for (unsigned j = 0; j < attr.elem_size; ++j)
                vertex_data.push_back(attr.data[i * attr.elem_size + j]);

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertex_data.size() * sizeof(float)), vertex_data.data(), GL_STATIC_DRAW);

    unsigned offset = 0;
    for (unsigned j = 0; j < attributes.size(); ++j) {
        glVertexAttribPointer(j, static_cast<GLint>(attributes[j].elem_size),
                              GL_FLOAT, GL_FALSE, stride * sizeof(float),
                              (void *) (offset * sizeof(float)));
        glEnableVertexAttribArray(j);
        offset += attributes[j].elem_size;
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glCheckError();
}

Geometry::Geometry(Geometry &&other) noexcept {
    *this = std::move(other);
}

Geometry &Geometry::operator=(Geometry &&other) noexcept {
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glCheckError();

    mode = other.mode;
    vao = other.vao;
    vbo = other.vbo;
    vertices_count = other.vertices_count;

    other.vao = 0;
    other.vbo = 0;

    return *this;
}

Geometry::~Geometry() {
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glCheckError();
}

void Geometry::draw() const {
    glBindVertexArray(vao);
    glDrawArrays(mode, 0, vertices_count);
    glBindVertexArray(0);
    glCheckError();
}

InstancedGeometry::InstancedGeometry(GLenum mode,
     const std::vector<VertexAttribute> &attributes,
     const std::vector<VertexAttribute> &instance_attributes)
     : InstancedGeometry{{mode, attributes}, attributes.size(), instance_attributes} {}

InstancedGeometry::InstancedGeometry(Geometry geom, std::size_t attribute_count,
                                     const std::vector<VertexAttribute> &instance_attributes)
    : Geometry{std::move(geom)},
      instance_count{static_cast<int>(instance_attributes[0].data.size() / instance_attributes[0].elem_size)} {

    int stride = 0;
    for (const auto &attr : instance_attributes)
        stride += static_cast<int>(attr.elem_size);

    std::vector<float> instance_data;
    instance_data.reserve(stride * instance_count);
    for (int i = 0; i < instance_count; ++i)
        for (const auto &attr : instance_attributes)
            for (unsigned j = 0; j < attr.elem_size; ++j)
                instance_data.push_back(attr.data[i * attr.elem_size + j]);

    glBindVertexArray(vao);
    glGenBuffers(1, &instance_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, instance_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(instance_data.size() * sizeof(float)), instance_data.data(), GL_DYNAMIC_DRAW);

    unsigned offset = 0;
    for (unsigned j = 0; j < instance_attributes.size(); ++j) {
        unsigned gl_idx = j + attribute_count;
        glVertexAttribPointer(gl_idx, static_cast<GLint>(instance_attributes[j].elem_size),
                              GL_FLOAT, GL_FALSE, stride * sizeof(float),
                              (void *) (offset * sizeof(float)));
        glEnableVertexAttribArray(gl_idx);
        glVertexAttribDivisor(gl_idx, 1);

        offset += instance_attributes[j].elem_size;
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
    glCheckError();
}

InstancedGeometry::~InstancedGeometry() {
    glDeleteBuffers(1, &instance_vbo);
    glCheckError();
}

void InstancedGeometry::draw() const {
    glBindVertexArray(vao);
    glDrawArraysInstanced(mode, 0, vertices_count, instance_count);
    glBindVertexArray(0);
    glCheckError();
}

void InstancedGeometry::update_instance_data(std::span<const float> instance_data) const {
    glBindBuffer(GL_ARRAY_BUFFER, instance_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(instance_data.size() * sizeof(float)), instance_data.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glCheckError();
}

namespace procedural {

Geometry triangle(bool color) {
    std::vector positions{
        0.0f,  0.5f, 0.0f,   // top vertex (red)
        -0.5f, -0.5f, 0.0f,  // bottom left (green)
        0.5f, -0.5f, 0.0f    // bottom right (blue)
    };
    std::vector tex_coords{
        0.5f, 1.f,   // top vertex (red)
        0.f, 0.f,  // bottom left (green)
        1.f, 0.f    // bottom right (blue)
    };
    std::vector<VertexAttribute> attributes{{3, positions}, {2, tex_coords}};

    if (color)
        attributes.emplace_back(3, std::vector{
            1.0f, 0.0f, 0.0f,  // top vertex (red)
            0.0f, 1.0f, 0.0f,  // bottom left (green)
            0.0f, 0.0f, 1.0f   // bottom right (blue)
        });

    return {GL_TRIANGLES, attributes};
}

Geometry quad(float side_length, bool tex_coord, bool color) {
    std::vector positions{
        -0.5f,  0.5f, 0.0f,
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f,
        0.5f,  0.5f, 0.0f,
        -0.5f, 0.5f, 0.0f,
        0.5f, -0.5f, 0.0f
    };
    for (float &x : positions)
        x *= side_length;
    std::vector<VertexAttribute> attributes{{3, positions}};

    if (tex_coord)
        attributes.emplace_back(2, std::vector{
            0.f, 1.f,
            0.f, 0.f,
            1.f, 0.f,
            1.f, 1.f,
            0.f, 1.f,
            1.f, 0.f,
        });
    if (color)
        attributes.emplace_back(3, std::vector{
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
        });

    return {GL_TRIANGLES, attributes};
}

Geometry axes(float half_size) {
    std::vector<float> positions{
        // x-axis
        -half_size, 0.0f, 0.0f,
        half_size, 0.0f, 0.0f,
        // y-axis
        0.0f, -half_size, 0.0f,
        0.0f, half_size, 0.0f,
        // z-axis
        0.0f, 0.0f, -half_size,
        0.0f, 0.0f, half_size,
        // diagonal xy
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        // diagonal xz
        1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        // diagonal yz
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };
    std::vector<float> colors{
        // x-axis (red)
        1.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        // y-axis (green)
        0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        // z-axis (blue)
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        // diagonal xy
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        // diagonal xz
        1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        // diagonal yz
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
    };

    return {GL_LINES, {{3, positions}, {3, colors}}};
}

}
