#version 450 core

in VertexData {
    vec2 centered_pos;
    vec3 center_view;
    flat uint p_id;
} in_data;

out vec4 frag_color;
layout (depth_less) out float gl_FragDepth;

layout(std140, binding = 0) uniform CameraData {
    mat4 projection;
    mat4 view;
    vec3 eye_position;
};

layout(std430, binding = 0) buffer VecVisualizerBuffer {
    vec4 vec_visualizer[];
};

layout(std430, binding = 1) buffer FloatVisualizerBuffer {
    float float_visualizer[];
};

uniform bool visualize_vec;
uniform bool visualize_float;

uniform bool norm;
uniform float min_value;
uniform float max_value;

const float RADIUS = 0.02;

void main() {
    vec3 n = vec3(in_data.centered_pos,
        1 - (in_data.centered_pos.x*in_data.centered_pos.x + in_data.centered_pos.y*in_data.centered_pos.y));
    if (n.z < 0.)
        discard;
    n.z = sqrt(n.z);

    vec3 view_pos = in_data.center_view + n * RADIUS;
    vec4 clip_pos = projection * vec4(view_pos, 1.);
    float ndc_depth = clip_pos.z / clip_pos.w;
    gl_FragDepth = ndc_depth * 0.5 + 0.5;

    vec3 base_color;
    if (visualize_vec)
        base_color = vec_visualizer[in_data.p_id].xyz;
    else if (visualize_float)
        base_color = vec3(float_visualizer[in_data.p_id]);
    else
        base_color = vec3(0., 0.5, 1.);

    if (norm)
        base_color = (base_color - vec3(min_value)) / (max_value - min_value);

    frag_color = vec4(base_color * dot(n, vec3(0., 0., 1.)), 1.);
}
