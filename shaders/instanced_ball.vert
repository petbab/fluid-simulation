#version 450 core

layout (location = 0) in vec3 vertex_position;
layout (location = 1) in vec3 instance_position;

uniform mat4 view;
uniform mat4 projection;
uniform vec3 camera_pos;

out VertexData {
    vec2 centered_pos;
    vec3 center_view;
} out_data;

const float RADIUS = 1.;

void main()
{
    // Calculate billboard orientation
    vec3 to_camera = normalize(camera_pos - instance_position);
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(up, to_camera));
    up = cross(to_camera, right);
    vec3 world_position = instance_position + (right * vertex_position.x + up * vertex_position.y) * RADIUS;

    gl_Position = projection * view * vec4(world_position, 1.0);

    out_data.centered_pos = vec2(vertex_position) * 2.;
    out_data.center_view = vec3(view * vec4(instance_position, 1.));
}
