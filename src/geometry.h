#pragma once

#include <vector>
#include <glad/glad.h>


class Geometry {
    static constexpr GLuint POSITION_LOCATION = 0,
                            TEX_COORD_LOCATION = 1,
                            COLOR_LOCATION = 2;

public:
    Geometry(GLenum mode,
        const std::vector<float> &positions,
        const std::vector<float> &tex_coords = {},
        const std::vector<float> &colors = {});

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

}
