#pragma once
#include <vector>
#include <glm/glm.hpp>


unsigned generate_boundary_particles(std::vector<glm::vec4>& positions,
                                     const std::vector<glm::vec3>& triangle_vertices,
                                     float particle_radius);
