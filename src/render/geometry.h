#pragma once

#include <vector>
#include <glad/glad.h>
#include <span>
#include <filesystem>


struct VertexAttribute {
    unsigned elem_size;
    std::span<const float> data;
};

class Geometry {
public:
    Geometry(GLenum mode, const std::vector<VertexAttribute> &attributes,
        std::span<unsigned> indices = {});

    static Geometry from_file(const std::filesystem::path& file_path, bool normalize_mesh = true);

    Geometry(const Geometry&) = delete;
    Geometry& operator=(const Geometry&) = delete;
    Geometry(Geometry&&) noexcept;
    Geometry& operator=(Geometry&&) noexcept;
    virtual ~Geometry();

    virtual void draw() const;
    unsigned get_vbo() const { return vbo; }

protected:
    GLenum mode;
    unsigned vbo = 0,
             vao = 0,
             ebo = 0;
    unsigned vertices_count, indices_count;
};

class InstancedGeometry : public Geometry {
public:
    InstancedGeometry(GLenum mode,
        const std::vector<VertexAttribute> &attributes,
        const std::vector<VertexAttribute> &instance_attributes,
        std::span<unsigned> indices = {});
    InstancedGeometry(Geometry geometry, std::size_t attribute_count,
        const std::vector<VertexAttribute> &instance_attributes);
    ~InstancedGeometry() override;

    void draw() const override;
    void update_instance_data(std::span<const float> instance_data) const;
    unsigned get_instance_vbo() const { return instance_vbo; }

private:
    unsigned instance_vbo = 0;
    unsigned instance_count;
};

namespace procedural {

Geometry triangle(bool color);

Geometry quad(float side_length, bool tex_coord, bool color);

Geometry axes(float half_size = 5.);

}
