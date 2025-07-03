#include "geometry.h"


Geometry::Geometry(GLenum mode,
    const std::vector<float> &positions,
    const std::vector<float> &tex_coords,
    const std::vector<float> &colors)
    : mode{mode}, vertices_count{static_cast<int>(positions.size() / 3)} {
    vertex_data.reserve(positions.size() + colors.size() + tex_coords.size());
    for (std::size_t i = 0; i < vertices_count; ++i) {
        vertex_data.push_back(positions[i * 3]);
        vertex_data.push_back(positions[i * 3 + 1]);
        vertex_data.push_back(positions[i * 3 + 2]);

        if (!tex_coords.empty()) {
            vertex_data.push_back(tex_coords[i * 2]);
            vertex_data.push_back(tex_coords[i * 2 + 1]);
        }
        if (!colors.empty()) {
            vertex_data.push_back(colors[i * 3]);
            vertex_data.push_back(colors[i * 3 + 1]);
            vertex_data.push_back(colors[i * 3 + 2]);
        }
    }

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertex_data.size() * sizeof(float)), vertex_data.data(), GL_STATIC_DRAW);

    GLsizei stride = (3 + (colors.empty() ? 0 : 3) + (tex_coords.empty() ? 0 : 2)) * sizeof(float);
    glVertexAttribPointer(POSITION_LOCATION, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(POSITION_LOCATION);

    if (!tex_coords.empty()) {
        glVertexAttribPointer(TEX_COORD_LOCATION, 2, GL_FLOAT, GL_FALSE, stride, (void *) (3 * sizeof(float)));
        glEnableVertexAttribArray(TEX_COORD_LOCATION);
    }
    if (!colors.empty()) {
        glVertexAttribPointer(COLOR_LOCATION, 3, GL_FLOAT, GL_FALSE, stride, (void *) (5 * sizeof(float)));
        glEnableVertexAttribArray(COLOR_LOCATION);
    }

    glBindVertexArray(0);
}

Geometry::~Geometry() {
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
}

void Geometry::draw() const {
    glBindVertexArray(VAO);
    glDrawArrays(mode, 0, vertices_count);
}

namespace procedural {

Geometry triangle(bool color) {
    // Triangle vertices with positions and colors
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
    std::vector<float> colors;
    if (color) {
        colors = {
            1.0f, 0.0f, 0.0f,  // top vertex (red)
            0.0f, 1.0f, 0.0f,  // bottom left (green)
            0.0f, 0.0f, 1.0f   // bottom right (blue)
        };
    }
    return {GL_TRIANGLES, positions, tex_coords, colors};
}

Geometry quad(float side_length, bool color) {
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

    std::vector tex_coords{
        0.f, 1.f,
        0.f, 0.f,
        1.f, 0.f,
        1.f,  1.f,
        0.f, 1.f,
        1.f, 0.f,
    };
    std::vector<float> colors;
    if (color) {
        colors = {
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 1.0f,
        };
    }
    return {GL_TRIANGLES, positions, tex_coords, colors};
}

}
