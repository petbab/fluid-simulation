#version 450 core

// ----------------------------------------------------------------------------
// Input Variables
// ----------------------------------------------------------------------------
layout (location = 0) in vec3 position;  // The vertex position.
layout (location = 1) in vec3 normal;	 // The vertex normal.

layout(std140, binding = 0) uniform CameraData {
    mat4 projection;
    mat4 view;
    vec3 eye_position;
};

// The UBO with the model data.
layout (std140, binding = 1) uniform ModelData {
    mat4 model;			// The model matrix.
    mat3 model_it;		// The inverse of the transpose of the top-left part 3x3 of the model matrix.
};

// ----------------------------------------------------------------------------
// Output Variables
// ----------------------------------------------------------------------------
out VertexData
{
    vec3 position_ws;	  // The vertex position in world space.
    vec3 normal_ws;		  // The vertex normal in world space.
} out_data;

// ----------------------------------------------------------------------------
// Main Method
// ----------------------------------------------------------------------------
void main()
{
    out_data.position_ws = vec3(model * vec4(position, 1.));
    out_data.normal_ws = -normalize(model_it * normal);

    gl_Position = projection * view * model * vec4(position, 1.);
}
