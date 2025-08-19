#pragma once

#include <vector>
#include <glad/glad.h>
#include <span>


struct VertexAttribute {
    unsigned elem_size;
    std::span<const float> data;
};

class Geometry {
public:
    Geometry(GLenum mode, const std::vector<VertexAttribute> &attributes);

    Geometry(const Geometry&) = delete;
    Geometry& operator=(const Geometry&) = delete;
    Geometry(Geometry&&) noexcept;
    Geometry& operator=(Geometry&&) noexcept;
    virtual ~Geometry();

    virtual void draw() const;

protected:
    GLenum mode;
    unsigned vbo = 0,
             vao = 0;
    int vertices_count;
};

class InstancedGeometry : public Geometry {
public:
    InstancedGeometry(GLenum mode,
        const std::vector<VertexAttribute> &attributes,
        const std::vector<VertexAttribute> &instance_attributes);
    InstancedGeometry(Geometry geometry, std::size_t attribute_count,
        const std::vector<VertexAttribute> &instance_attributes);

    ~InstancedGeometry() override;

    void draw() const override;

    void update_instance_data(std::span<const float> instance_data) const;

private:
    unsigned instance_vbo = 0;
    int instance_count;
};

namespace procedural {

Geometry triangle(bool color);

Geometry quad(float side_length, bool tex_coord, bool color);

Geometry axes(float half_size = 5.);

}
