#version 450 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 vertexColor;

layout(std140, binding = 0) uniform CameraData {
    mat4 projection;
    mat4 view;
    vec3 eye_position;
};

void main() {
    gl_Position = projection * view * vec4(aPos, 1.0);
    vertexColor = aColor;
}
