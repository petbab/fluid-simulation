#version 450 core

//----------------------------------------------------------------------------
// Input Variables
// ----------------------------------------------------------------------------
in VertexData
{
    vec2 texture_coords;
} in_data;

layout(binding = 0) uniform sampler2D tex;

uniform bool color;
uniform bool depth;

const float NEAR = 0.1f;
const float FAR = 100.f;

// ----------------------------------------------------------------------------
// Output Variables
// ----------------------------------------------------------------------------
// The final output color.
layout (location = 0) out vec4 final_color;

// ----------------------------------------------------------------------------
// Main Method
// ----------------------------------------------------------------------------
void main()
{
    if (depth) {
        float d = texture(tex, in_data.texture_coords).r;
        float z = (2.0 * NEAR * FAR) / (FAR + NEAR - (d * 2.0 - 1.0) * (FAR - NEAR));
        final_color = vec4(vec3(z / FAR), 1.0);
    } else if (color) {
        final_color = vec4(texture(tex, in_data.texture_coords).rgb, 1.);
    } else
        final_color = vec4(texture(tex, in_data.texture_coords).rrr, 1.);
}
