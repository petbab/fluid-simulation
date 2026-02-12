#include "boundary.h"
#include <open3d/Open3D.h>
#include <cassert>


static constexpr float SPACING_MULT = 0.5f;

unsigned generate_boundary_particles(
    std::vector<glm::vec3>& positions,
    const std::vector<glm::vec3>& triangle_vertices,
    float particle_radius
) {
    assert(triangle_vertices.size() % 3 == 0);

    unsigned count_start = positions.size();

    open3d::geometry::TriangleMesh mesh{};
    mesh.vertices_.reserve(triangle_vertices.size());
    for (const glm::vec3 &v : triangle_vertices)
        mesh.vertices_.emplace_back(v.x, v.y, v.z);

    unsigned num_triangles = triangle_vertices.size() / 3;
    mesh.triangles_.reserve(num_triangles);
    for (unsigned i = 0; i < num_triangles; i++)
        mesh.triangles_.emplace_back(i * 3, i * 3 + 1, i * 3 + 2);

    auto voxel_grid = open3d::geometry::VoxelGrid::CreateFromTriangleMesh(
        mesh, 2.f * particle_radius * SPACING_MULT);
    for (const open3d::geometry::Voxel &v : voxel_grid->GetVoxels()) {
        auto center = voxel_grid->GetVoxelCenterCoordinate(v.grid_index_);
        positions.emplace_back(center[0], center[1], center[2]);
    }

    return positions.size() - count_start;
}
