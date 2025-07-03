#version 450 core
out vec4 FragColor;
in vec3 vertexColor;
in vec2 texCoord;

void main() {
    vec2 centered_tex_coord = texCoord * 2 - vec2(1, 1);
    if (centered_tex_coord.x*centered_tex_coord.x + centered_tex_coord.y*centered_tex_coord.y > 1)
        discard;
    FragColor = vec4(vertexColor, 1.0);
}
