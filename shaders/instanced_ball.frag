#version 450 core

in VertexData {
    vec2 centered_pos;
    vec3 center_view;
    flat bool is_boundary;
} in_data;

out vec4 frag_color;
layout (depth_less) out float gl_FragDepth;

layout(std140, binding = 0) uniform CameraData {
    mat4 projection;
    mat4 view;
    vec3 eye_position;
};

uniform bool show_boundary;

const float RADIUS = 0.02;
const vec3 FLUID_COLOR = vec3(0., 0.5, 1.);
const vec3 BOUNDARY_COLOR = vec3(1., 0.5, 0.);

void main() {
    if (in_data.is_boundary && !show_boundary)
        discard;

    vec3 n = vec3(in_data.centered_pos,
        1 - (in_data.centered_pos.x*in_data.centered_pos.x + in_data.centered_pos.y*in_data.centered_pos.y));
    if (n.z < 0.)
        discard;
    n.z = sqrt(n.z);

    vec3 view_pos = in_data.center_view + n * RADIUS;
    vec4 clip_pos = projection * vec4(view_pos, 1.);
    float ndc_depth = clip_pos.z / clip_pos.w;
    gl_FragDepth = ndc_depth * 0.5 + 0.5;

    vec3 color = in_data.is_boundary ? BOUNDARY_COLOR : FLUID_COLOR;
    frag_color = vec4(color * dot(n, vec3(0., 0., 1.)), 1.);
}
