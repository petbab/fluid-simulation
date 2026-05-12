#pragma once

#include <vector>
#include <glad/glad.h>
#include <span>
#include <filesystem>
#include <glm/glm.hpp>


/**
 * @brief Description of a single vertex attribute stream.
 */
struct VertexAttribute {
    unsigned elem_size;          ///< Number of float components per vertex (e.g. 3 for vec3).
    std::span<const float> data; ///< Span of raw float data.
};

/**
 * @brief OpenGL geometry buffer manager (VAO/VBO/EBO).
 *
 * Handles creation, destruction, and drawing of a geometry.
 * Supports indexed and non-indexed drawing modes.
 */
class Geometry {
public:
    /**
     * @brief Constructs geometry from vertex attributes and optional indices.
     * @param mode OpenGL draw mode (e.g. GL_TRIANGLES, GL_LINES).
     * @param attributes List of vertex attributes.
     * @param indices Optional index buffer data.
     */
    Geometry(GLenum mode, const std::vector<VertexAttribute> &attributes,
        std::span<unsigned> indices = {});

    /**
     * @brief Loads geometry from an OBJ file.
     * @param file_path Path to the .obj file.
     * @param normalize_mesh If true, normalizes the mesh to unit bounds.
     * @return Geometry loaded from the file.
     */
    static Geometry from_file(const std::filesystem::path& file_path, bool normalize_mesh = true);

    Geometry(const Geometry&) = delete;
    Geometry& operator=(const Geometry&) = delete;
    Geometry(Geometry&&) noexcept;
    Geometry& operator=(Geometry&&) noexcept;
    virtual ~Geometry();

    /** @brief Issues the OpenGL draw call for this geometry. */
    virtual void draw() const;

    /** @return The OpenGL VBO identifier. */
    unsigned get_vbo() const { return vbo; }

    /**
     * @brief Loads triangle vertices into a vector, transformed by a model matrix.
     * @param triangle_vertices Output vector for transformed vertices.
     * @param model Transformation matrix to apply.
     */
    void load_triangles(std::vector<glm::vec3> &triangle_vertices, const glm::mat4 &model = {1.f}) const;

protected:
    GLenum mode;              ///< OpenGL draw mode.
    unsigned vbo = 0,         ///< Vertex buffer object.
             vao = 0,         ///< Vertex array object.
             ebo = 0;         ///< Element buffer object (0 if unused).
    unsigned vertices_count;  ///< Number of vertices.
    unsigned indices_count;   ///< Number of indices (0 if unused).
    unsigned stride;          ///< Number of floats per vertex.
};

/**
 * @brief Geometry with instanced vertex attributes.
 *
 * Extends Geometry with an additional instance VBO for per-instance data,
 * enabling efficient rendering of many copies with a single draw call.
 */
class InstancedGeometry : public Geometry {
public:
    /**
     * @brief Constructs instanced geometry from scratch.
     * @param mode OpenGL draw mode.
     * @param attributes Per-vertex attributes.
     * @param instance_attributes Per-instance attributes.
     * @param indices Optional index buffer.
     */
    InstancedGeometry(GLenum mode,
        const std::vector<VertexAttribute> &attributes,
        const std::vector<VertexAttribute> &instance_attributes,
        std::span<unsigned> indices = {});

    /**
     * @brief Constructs instanced geometry from an existing Geometry.
     * @param geometry Base geometry to take ownership of.
     * @param attribute_count Number of per-vertex attributes already in the geometry.
     * @param instance_attributes Per-instance attributes to add.
     */
    InstancedGeometry(Geometry geometry, std::size_t attribute_count,
        const std::vector<VertexAttribute> &instance_attributes);
    ~InstancedGeometry() override;

    void draw() const override;

    /**
     * @brief Updates the instance attribute data on the GPU.
     * @param instance_data New instance data (must match original size).
     */
    void update_instance_data(std::span<const float> instance_data) const;

    /** @return The OpenGL instance VBO identifier. */
    unsigned get_instance_vbo() const { return instance_vbo; }

private:
    unsigned instance_vbo = 0;  ///< Instance vertex buffer object.
    unsigned instance_count;    ///< Number of instances.
    unsigned instance_stride;   ///< Number of floats per instance.
};

/**
 * @brief Procedural geometry generators.
 */
namespace procedural {

/**
 * @brief Creates a simple colored triangle.
 * @param color If true, includes per-vertex color attributes.
 * @return Triangle geometry.
 */
Geometry triangle(bool color);

/**
 * @brief Creates a quad.
 * @param side_length Length of each side.
 * @param tex_coord If true, includes texture coordinate attributes.
 * @param color If true, includes per-vertex color attributes.
 * @return Quad geometry.
 */
Geometry quad(float side_length, bool tex_coord, bool color);

/**
 * @brief Creates coordinate axes lines.
 * @param half_size Half-length of each axis line.
 * @return Axes geometry.
 */
Geometry axes(float half_size = 5.);

/**
 * @brief Creates a cube.
 * @param half_size Half-extents of the cube.
 * @return Cube geometry.
 */
Geometry cube(glm::vec3 half_size);

}
