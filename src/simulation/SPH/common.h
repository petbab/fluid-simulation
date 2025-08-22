#pragma once

#include <CompactNSearch>
#include <memory>
#include "../fluid_simulator.h"
#include "kernel.h"


class SPHBase : public FluidSimulator {
public:
    SPHBase(unsigned grid_count, BoundingBox bounding_box, float support_radius, bool is_2d = false)
        : FluidSimulator{grid_count, bounding_box, is_2d},
          n_search{std::make_unique<CompactNSearch::NeighborhoodSearch>(support_radius)} {
        densities.resize(positions.size());

        point_set_index = n_search->add_point_set(reinterpret_cast<float*>(positions.data()), positions.size());
        n_search->z_sort();
        n_search->find_neighbors();
    }

protected:
    void find_neighbors() { n_search->find_neighbors(); }
    void z_sort() { n_search->z_sort(); }

    void compute_densities(float particle_mass, const Kernel &kernel) {
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < densities.size(); ++i) {
            float density = kernel.W(glm::vec3{0.f});
            glm::vec3 xi = positions[i];

            for_neighbors(i, [&](unsigned j){
                density += kernel.W(xi - positions[j]);
            });

            densities[i] = density * particle_mass;
        }
    }

    void for_neighbors(unsigned i, auto f) {
        CompactNSearch::PointSet &ps = n_search->point_set(point_set_index);
        for (unsigned j = 0; j < ps.n_neighbors(point_set_index, i); ++j)
            f(ps.neighbor(point_set_index, i, j));
    }

    std::vector<float> densities;
private:
    std::unique_ptr<CompactNSearch::NeighborhoodSearch> n_search;
    unsigned point_set_index;
};
