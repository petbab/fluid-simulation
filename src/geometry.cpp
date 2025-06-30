#include "geometry.h"


Geometry::Geometry(GLenum mode, const std::vector<float> &positions, const std::vector<float> &colors)
    : mode{mode}, vertices_count{static_cast<int>(positions.size() / 3)} {
    vertex_data.reserve(positions.size() + colors.size());
    for (std::size_t i = 0; i < positions.size() - 2; i += 3) {
        vertex_data.push_back(positions[i]);
        vertex_data.push_back(positions[i + 1]);
        vertex_data.push_back(positions[i + 2]);

        if (!colors.empty()) {
            vertex_data.push_back(colors[i]);
            vertex_data.push_back(colors[i + 1]);
            vertex_data.push_back(colors[i + 2]);
        }
    }

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertex_data.size() * sizeof(float)), vertex_data.data(), GL_STATIC_DRAW);

    GLsizei stride = (3 + (colors.empty() ? 0 : 3)) * sizeof(float);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(0);

    if (!colors.empty()) {
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void *) (3 * sizeof(float)));
        glEnableVertexAttribArray(1);
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

Geometry triangle() {
    // Triangle vertices with positions and colors
    std::vector positions{
        0.0f,  0.5f, 0.0f,   // top vertex (red)
        -0.5f, -0.5f, 0.0f,  // bottom left (green)
        0.5f, -0.5f, 0.0f    // bottom right (blue)
    };
    std::vector colors{
        1.0f, 0.0f, 0.0f,  // top vertex (red)
        0.0f, 1.0f, 0.0f,  // bottom left (green)
        0.0f, 0.0f, 1.0f   // bottom right (blue)
    };
    return {GL_TRIANGLES, positions, colors};
}

}
