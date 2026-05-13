#pragma once
#include <vector>
#include <glm/glm.hpp>


/**
 * @brief Generates boundary particles from triangle mesh geometry.
 *
 * Samples the surface of the given triangle mesh with particles of
 * the specified radius and appends them to the positions vector.
 * @param positions Output vector for generated boundary particle positions.
 * @param triangle_vertices Triangle vertices defining the mesh surface.
 * @param particle_radius Radius of each boundary particle.
 * @return Number of boundary particles generated.
 */
unsigned generate_boundary_particles(std::vector<glm::vec4>& positions,
                                     const std::vector<glm::vec3>& triangle_vertices,
                                     float particle_radius);
