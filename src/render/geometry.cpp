#include <cassert>
#include "geometry.h"
#include "../debug.h"
#include "../tiny_obj_loader.h"


Geometry::Geometry(GLenum mode, const std::vector<VertexAttribute>& attributes, std::span<unsigned> indices)
    : mode{mode},
      vertices_count{static_cast<unsigned>(attributes[0].data.size() / attributes[0].elem_size)},
      indices_count{static_cast<unsigned>(indices.size())} {
    assert(!attributes.empty());

    stride = 0;
    for (const auto &attr : attributes)
        stride += static_cast<int>(attr.elem_size);

    std::vector<float> vertex_data;
    vertex_data.reserve(stride * vertices_count);
    for (int i = 0; i < vertices_count; ++i)
        for (const auto &attr : attributes)
            for (unsigned j = 0; j < attr.elem_size; ++j)
                vertex_data.push_back(attr.data[i * attr.elem_size + j]);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertex_data.size() * sizeof(float)), vertex_data.data(), GL_STATIC_DRAW);

    if (indices_count > 0) {
        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned), indices.data(), GL_STATIC_DRAW);
    }

    unsigned offset = 0;
    for (unsigned j = 0; j < attributes.size(); ++j) {
        glVertexAttribPointer(j, static_cast<GLint>(attributes[j].elem_size),
                              GL_FLOAT, GL_FALSE, stride * sizeof(float),
                              (void *) (offset * sizeof(float)));
        glEnableVertexAttribArray(j);
        offset += attributes[j].elem_size;
    }

    glBindVertexArray(0);
    glCheckError();
}

Geometry Geometry::from_file(const std::filesystem::path& file_path, bool normalize_mesh) {
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(file_path)) {
        std::string err = reader.Error().empty() ? "" : (": " + reader.Error());
        throw std::runtime_error{"TinyObjReader: failed to load " + file_path.string() + err};
    }

    if (!reader.Warning().empty()) {
        std::cerr << "TinyObjReader: " << reader.Warning();
    }

    // Use only the first shape
    // TODO: more shapes?
    const tinyobj::shape_t& shape = reader.GetShapes()[0];
    const tinyobj::attrib_t& attrib = reader.GetAttrib();

    std::vector<float> positions, normals, tex_coords;
    positions.reserve(shape.mesh.indices.size() * 3);
    normals.reserve(shape.mesh.indices.size() * 3);
    tex_coords.reserve(shape.mesh.indices.size() * 2);

    glm::vec3 min{std::numeric_limits<float>::max()};
    glm::vec3 max{std::numeric_limits<float>::min()};

    // Index offset for the current shape
    unsigned index_offset = 0;

    // Iterate over all faces in the shape
    for (unsigned f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
        // Get the number of vertices for this face
        unsigned fv = shape.mesh.num_face_vertices[f];

        if (fv > 3)
            throw std::runtime_error{"In " + file_path.string() + ": face " + std::to_string(f) + " has "
                + std::to_string(fv) + " vertices (not a triangle)"};

        // Iterate over the 3 vertices of the triangle
        for (unsigned v = 0; v < fv; v++) {
            // Access to vertex indices
            tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

            // Get vertex position
            float vx = attrib.vertices[3 * idx.vertex_index];
            float vy = attrib.vertices[3 * idx.vertex_index + 1];
            float vz = attrib.vertices[3 * idx.vertex_index + 2];

            positions.insert(positions.end(), {vx, vy, vz});

            if (normalize_mesh) {
                min.x = std::min(min.x, vx);
                min.y = std::min(min.y, vy);
                min.z = std::min(min.z, vz);
                max.x = std::max(max.x, vx);
                max.y = std::max(max.y, vy);
                max.z = std::max(max.z, vz);
            }

            // Get normal if available
            float nx = 0, ny = 0, nz = 0;
            if (idx.normal_index >= 0) {
                nx = attrib.normals[3 * idx.normal_index];
                ny = attrib.normals[3 * idx.normal_index + 1];
                nz = attrib.normals[3 * idx.normal_index + 2];
            }
            normals.insert(normals.end(), {nx, ny, nz});

            // Get texture coordinate if available
            float tx = 0, ty = 0;
            if (idx.texcoord_index >= 0) {
                tx = attrib.texcoords[2 * idx.texcoord_index];
                ty = attrib.texcoords[2 * idx.texcoord_index + 1];
            }
            tex_coords.insert(tex_coords.end(), {tx, ty});
        }

        index_offset += fv;
    }

    if (normalize_mesh) {
        glm::vec3 center = (min + max) * 0.5f;
        glm::vec3 diff = max - min;
        float max_diff = std::max(std::max(diff.x, diff.y), diff.z);
        for (int i = 0; i < positions.size(); i += 3) {
            positions[i] = (positions[i] - center.x) / max_diff;
            positions[i + 1] = (positions[i + 1] - center.y) / max_diff;
            positions[i + 2] = (positions[i + 2] - center.z) / max_diff;
        }
    }

    return {GL_TRIANGLES, {{3, positions}, {3, normals}, {2, tex_coords}}};
}

Geometry::Geometry(Geometry &&other) noexcept {
    *this = std::move(other);
}

Geometry &Geometry::operator=(Geometry &&other) noexcept {
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glCheckError();

    mode = other.mode;
    vao = other.vao;
    vbo = other.vbo;
    ebo = other.ebo;

    vertices_count = other.vertices_count;
    indices_count = other.indices_count;
    stride = other.stride;

    other.vao = 0;
    other.vbo = 0;
    other.ebo = 0;

    return *this;
}

Geometry::~Geometry() {
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);
    glCheckError();
}

void Geometry::draw() const {
    glBindVertexArray(vao);
    if (indices_count > 0)
        glDrawElements(mode, indices_count, GL_UNSIGNED_INT, nullptr);
    else
        glDrawArrays(mode, 0, vertices_count);
    glBindVertexArray(0);
    glCheckError();
}

void Geometry::load_triangles(std::vector<glm::vec3> &triangle_vertices, const glm::mat4 &model) const {
    assert(mode == GL_TRIANGLES);
    assert(ebo == 0 && indices_count == 0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // Map the buffer to CPU memory
    const float* buffer = static_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY));
    glCheckError();
    if (buffer == nullptr)
        throw std::runtime_error{"Failed to map triangle VBO."};

    unsigned offset = 0;
    triangle_vertices.resize(vertices_count);
    for (glm::vec3 &v : triangle_vertices) {
        v = model * glm::vec4(buffer[offset], buffer[offset + 1], buffer[offset + 2], 1.0f);
        offset += stride;
    }

    // Unmap when done
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

InstancedGeometry::InstancedGeometry(GLenum mode,
     const std::vector<VertexAttribute> &attributes,
     const std::vector<VertexAttribute> &instance_attributes,
     std::span<unsigned> indices)
     : InstancedGeometry{{mode, attributes, indices}, attributes.size(), instance_attributes} {}

InstancedGeometry::InstancedGeometry(Geometry geom, std::size_t attribute_count,
                                     const std::vector<VertexAttribute> &instance_attributes)
    : Geometry{std::move(geom)},
      instance_count{static_cast<unsigned>(instance_attributes[0].data.size() / instance_attributes[0].elem_size)} {

    instance_stride = 0;
    for (const auto &attr : instance_attributes)
        instance_stride += static_cast<int>(attr.elem_size);

    std::vector<float> instance_data;
    instance_data.reserve(instance_stride * instance_count);
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
                              GL_FLOAT, GL_FALSE, instance_stride * sizeof(float),
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
    if (indices_count > 0)
        glDrawElementsInstanced(mode, indices_count, GL_UNSIGNED_INT, nullptr, instance_count);
    else
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
