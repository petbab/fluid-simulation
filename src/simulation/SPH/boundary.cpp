#include "boundary.h"

#include <algorithm>
#include <cmath>
#include <glm/glm.hpp>
#include <ranges>


struct Triangle {
    glm::vec3 a, b, c;

    glm::vec3 sample(float u, float v) const {
        // Barycentric interpolation: P = (1-u-v)*v0 + u*v1 + v*v2
        float w = 1.0f - u - v;
        return a * w + b * u + c * v;
    }
};


// bool pointInTriangle(
//     const glm::vec2& p,
//     const glm::vec2& a,
//     const glm::vec2& b,
//     const glm::vec2& c,
//     double eps = 1e-12
// ) {
//     glm::vec2 v0{b.x - a.x, b.y - a.y};
//     glm::vec2 v1{c.x - a.x, c.y - a.y};
//     glm::vec2 v2{p.x - a.x, p.y - a.y};
//
//     double d00 = v0.x*v0.x + v0.y*v0.y;
//     double d01 = v0.x*v1.x + v0.y*v1.y;
//     double d11 = v1.x*v1.x + v1.y*v1.y;
//     double d20 = v2.x*v0.x + v2.y*v0.y;
//     double d21 = v2.x*v1.x + v2.y*v1.y;
//
//     double denom = d00*d11 - d01*d01;
//     if (std::abs(denom) < eps) return false;
//
//     double u = (d11*d20 - d01*d21) / denom;
//     double v = (d00*d21 - d01*d20) / denom;
//
//     return u >= -eps && v >= -eps && (u + v) <= 1.0 + eps;
// }
//
// static void generate_boundary_particles(std::vector<glm::vec3>& positions, const Triangle& tri, float particle_radius) {
//     const float h = particle_radius * 2.f;
//
//     glm::vec3 e1 = tri.b - tri.a;
//     glm::vec3 e2 = tri.c - tri.a;
//
//     glm::vec3 n = cross(e1, e2);
//     double area2 = glm::length(n);
//     if (area2 < 1e-12)
//         throw std::runtime_error("Degenerate triangle");
//
//     glm::vec3 xHat = glm::normalize(e1);
//     glm::vec3 nHat = glm::normalize(n);
//     glm::vec3 yHat = cross(nHat, xHat);
//
//     // Triangle in 2D
//     glm::vec2 p0{0.0, 0.0};
//     glm::vec2 p1{dot(e1, xHat), dot(e1, yHat)};
//     glm::vec2 p2{dot(e2, xHat), dot(e2, yHat)};
//
//     // Bounding box
//     double minX = std::fmin(p0.x, std::fmin(p1.x, p2.x)) - h;
//     double maxX = std::fmax(p0.x, std::fmax(p1.x, p2.x)) + h;
//     double minY = std::fmin(p0.y, std::fmin(p1.y, p2.y)) - h;
//     double maxY = std::fmax(p0.y, std::fmax(p1.y, p2.y)) + h;
//
//     // Triangular lattice
//     glm::vec2 a1{h, 0.0};
//     glm::vec2 a2{0.5*h, std::sqrt(3.0)*h*0.5};
//
//     int iMin = (int)std::floor(minX / h) - 1;
//     int iMax = (int)std::ceil (maxX / h) + 1;
//     int jMin = (int)std::floor(minY / a2.y) - 1;
//     int jMax = (int)std::ceil (maxY / a2.y) + 1;
//
//     for (int i = iMin; i <= iMax; ++i) {
//         for (int j = jMin; j <= jMax; ++j) {
//             glm::vec2 p = a1 * static_cast<float>(i) + a2 * static_cast<float>(j);
//
//             if (p.x < minX || p.x > maxX || p.y < minY || p.y > maxY)
//                 continue;
//
//             if (pointInTriangle(p, p0, p1, p2)) {
//                 glm::vec3 world =
//                     tri.a +
//                     xHat * p.x +
//                     yHat * p.y;
//                 positions.push_back(world);
//             }
//         }
//     }
// }

static void generate_boundary_particles(std::vector<glm::vec3>& positions, const Triangle& tri, float particle_radius) {
    const glm::vec3 edge1 = tri.b - tri.a;
    const glm::vec3 edge2 = tri.c - tri.a;

    float edge1_len = glm::length(edge1);
    float edge2_len = glm::length(edge2);

    if (edge1_len < 1e-6f || edge2_len < 1e-6f) {
        // Degenerate triangle, skip
        return;
    }

    // Determine sampling resolution
    int samplesU = static_cast<int>(std::ceil(edge1_len / (2.f * particle_radius)));
    int samplesV = static_cast<int>(std::ceil(edge2_len / (2.f * particle_radius)));

    // Sample points on the triangle using barycentric coordinates
    for (int i = 0; i <= samplesU; ++i) {
        for (int j = 0; j <= samplesV; ++j) {
            float u = static_cast<float>(i) / samplesU;
            float v = static_cast<float>(j) / samplesV;

            // Check if point is inside triangle (u + v <= 1)
            if (u + v <= 1.0f) {
                glm::vec3 pos = tri.sample(u, v);
                positions.push_back(pos);
            }
        }
    }
}

unsigned generate_boundary_particles(
    std::vector<glm::vec3>& positions,
    const std::vector<glm::vec3>& triangle_vertices,
    float particle_radius
) {
    unsigned count_start = positions.size();

    for (int i = 0; i < triangle_vertices.size(); i += 3) {
        Triangle tri{triangle_vertices[i], triangle_vertices[i + 1], triangle_vertices[i + 2]};
        generate_boundary_particles(positions, tri, particle_radius);
    }

    return positions.size() - count_start;
}
