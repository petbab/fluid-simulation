#version 450 core

in VertexData {
    vec2 centered_pos;
    vec3 center_view;
} in_data;

out vec4 frag_color;
layout (depth_less) out float gl_FragDepth;

layout(std140, binding = 0) uniform CameraData {
    mat4 projection;
    mat4 view;
    vec3 position;
} camera;

const float RADIUS = 0.1;

void main() {
    vec3 n = vec3(in_data.centered_pos,
        1 - (in_data.centered_pos.x*in_data.centered_pos.x + in_data.centered_pos.y*in_data.centered_pos.y));
    if (n.z < 0.)
        discard;
    n.z = sqrt(n.z);

    vec3 view_pos = in_data.center_view + n * RADIUS;
    vec4 clip_pos = camera.projection * vec4(view_pos, 1.);
    float ndc_depth = clip_pos.z / clip_pos.w;
    gl_FragDepth = ndc_depth * 0.5 + 0.5;

    frag_color = vec4(0.1, 0.3, 0.9, 1.);
}
