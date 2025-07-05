#version 450 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec3 aColor;

out vec3 vertexColor;
out vec2 texCoord;

uniform mat4 view;
uniform mat4 projection;

void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
    vertexColor = vec3(0, .2, 1);
    texCoord = aTexCoord;
}
