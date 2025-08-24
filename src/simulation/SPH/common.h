#pragma once

#include <CompactNSearch>
#include <memory>
#include "../fluid_simulator.h"
#include "kernel.h"


class SPHBase : public FluidSimulator {
    static constexpr float ELASTICITY = 0.9f;
    static constexpr float XSPH_ALPHA = 0.01f;

public:
    SPHBase(unsigned grid_count, BoundingBox bounding_box, float support_radius, bool is_2d = false);

protected:
    void find_neighbors() { n_search->find_neighbors(); }
    void z_sort();

    void compute_densities(float particle_mass, const Kernel &kernel);

    void resolve_collisions();

    void for_neighbors(unsigned i, auto f) {
        CompactNSearch::PointSet &ps = n_search->point_set(point_set_index);
        for (unsigned j = 0; j < ps.n_neighbors(point_set_index, i); ++j)
            f(ps.neighbor(point_set_index, i, j));
    }

    void apply_XSPH(const Kernel &kernel, float particle_mass);

    void reset() override;

    std::vector<glm::vec3> velocities, XSPH_accel;
    std::vector<float> densities;
private:
    std::unique_ptr<CompactNSearch::NeighborhoodSearch> n_search;
    unsigned point_set_index;
};
