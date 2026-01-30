#include <gtest/gtest.h>
#include "../../src/cuda/nsearch/nsearch.h"
#include <thrust/device_vector.h>
#include <array>
#include "../../src/cuda/init.h"


TEST(NSearch, Constructor) {
    NSearchWrapper n_search_wrapper{1.f};
}

TEST(NSearch, CopyFromDevice) {
    constexpr float cell_size = 1.f;
    NSearchWrapper n_search_wrapper{cell_size};
    NSearchHost h_n_search = NSearchHost::copy_from_device(n_search_wrapper.dev_ptr());

    EXPECT_FLOAT_EQ(cell_size, h_n_search.cell_size);
}

// TEST(NSearch, SingleCell) {
//     NSearchWrapper n_search_wrapper{1.f};
//
//     constexpr unsigned particles = 100;
//     thrust::device_vector<float> positions(particles * 3, 0.f);
//     n_search_wrapper.rebuild(thrust::raw_pointer_cast(positions.data()), particles);
//
//     NSearchHost h_n_search = NSearchHost::copy_from_device(n_search_wrapper.dev_ptr());
//
//     // Expect one cell
//     auto t_it = std::find_if_not(h_n_search.table.begin(), h_n_search.table.end(),
//         [](auto h){ return h == NSearch::EMPTY_HASH; });
//     ASSERT_NE(t_it, h_n_search.table.end());
//
//     // No other cells
//     EXPECT_EQ(std::find_if_not(std::next(t_it), h_n_search.table.end(),
//         [](auto h){ return h == NSearch::EMPTY_HASH; }), h_n_search.table.end());
//
//     // Expect one cell with all 100 particles
//     unsigned indices_start = t_it - h_n_search.table.begin();
//     for (unsigned i = 0; i < NSearch::MAX_PARTICLES_IN_CELL; ++i) {
//         unsigned idx = h_n_search.particle_indices[NSearch::MAX_PARTICLES_IN_CELL * indices_start + i];
//
//         if (i < particles) {
//             EXPECT_NE(idx, NSearch::EMPTY_IDX);
//             EXPECT_LT(idx, particles);
//         } else {
//             EXPECT_EQ(idx, NSearch::EMPTY_IDX);
//         }
//     }
// }
//
// static NSearch fake_dev_n_search(NSearchHost &h_n_search) {
//     return {h_n_search.table.data(), h_n_search.particle_indices.data(), h_n_search.cell_size};
// }
//
// TEST(NSearch, SingleCellListNeighbors) {
//     NSearchWrapper n_search_wrapper{1.f};
//
//     constexpr unsigned particles = 100;
//     thrust::device_vector<float> positions(particles * 3, 0.f);
//     n_search_wrapper.rebuild(thrust::raw_pointer_cast(positions.data()), particles);
//
//     NSearchHost h_n_search = NSearchHost::copy_from_device(n_search_wrapper.dev_ptr());
//     NSearch fd_n_search = fake_dev_n_search(h_n_search);
//
//     std::array<unsigned, 27 * NSearch::MAX_PARTICLES_IN_CELL> neighbors{};
//     unsigned nbr_len = fd_n_search.list_neighbors(glm::vec3{0.}, neighbors.data());
//     EXPECT_EQ(nbr_len, particles);
//
//     for (unsigned i = 0; i < nbr_len; ++i) {
//         unsigned idx = neighbors[i];
//         EXPECT_NE(idx, NSearch::EMPTY_IDX);
//         EXPECT_LT(idx, particles);
//     }
// }
//
// TEST(NSearch, TwoCellListNeighbors) {
//     NSearchWrapper n_search_wrapper{1.f};
//
//     constexpr unsigned particles = 100;
//     thrust::device_vector<float> positions(particles * 3);
//     thrust::fill_n(positions.begin(), 150, 0.1f);
//     thrust::fill_n(positions.begin() + 150, 150, 1.1f);
//
//     n_search_wrapper.rebuild(thrust::raw_pointer_cast(positions.data()), particles);
//
//     NSearchHost h_n_search = NSearchHost::copy_from_device(n_search_wrapper.dev_ptr());
//
//     // Expect two cells
//     auto fst_it = std::find_if_not(h_n_search.table.begin(), h_n_search.table.end(),
//         [](auto h){ return h == NSearch::EMPTY_HASH; });
//     ASSERT_NE(fst_it, h_n_search.table.end());
//     auto snd_it = std::find_if_not(std::next(fst_it), h_n_search.table.end(),
//         [](auto h){ return h == NSearch::EMPTY_HASH; });
//     ASSERT_NE(snd_it, h_n_search.table.end());
//
//     // No other cells
//     EXPECT_EQ(std::find_if_not(std::next(snd_it), h_n_search.table.end(),
//         [](auto h){ return h == NSearch::EMPTY_HASH; }), h_n_search.table.end());
//
//     NSearch fd_n_search = fake_dev_n_search(h_n_search);
//
//     std::array<unsigned, 27 * NSearch::MAX_PARTICLES_IN_CELL> neighbors{};
//     for (auto pos : {glm::vec3{.1f}, glm::vec3{1.1f}}) {
//         unsigned nbr_len = fd_n_search.list_neighbors(pos, neighbors.data());
//         EXPECT_EQ(nbr_len, particles);
//
//         for (unsigned i = 0; i < nbr_len; ++i) {
//             unsigned idx = neighbors[i];
//             EXPECT_NE(idx, NSearch::EMPTY_IDX);
//             EXPECT_LT(idx, particles);
//         }
//     }
// }
//
// TEST(NSearch, ThreeCellListNeighbors) {
//     NSearchWrapper n_search_wrapper{1.f};
//
//     constexpr unsigned particles = 90;
//     thrust::device_vector<float> positions(particles * 3);
//     thrust::fill_n(positions.begin(), 90, 0.1f);
//     thrust::fill_n(positions.begin() + 90, 90, 1.1f);
//     thrust::fill_n(positions.begin() + 180, 90, 2.1f);
//
//     n_search_wrapper.rebuild(thrust::raw_pointer_cast(positions.data()), particles);
//
//     NSearchHost h_n_search = NSearchHost::copy_from_device(n_search_wrapper.dev_ptr());
//
//     // Expect three cells
//     auto fst_it = std::find_if_not(h_n_search.table.begin(), h_n_search.table.end(),
//         [](auto h){ return h == NSearch::EMPTY_HASH; });
//     ASSERT_NE(fst_it, h_n_search.table.end());
//     auto snd_it = std::find_if_not(std::next(fst_it), h_n_search.table.end(),
//         [](auto h){ return h == NSearch::EMPTY_HASH; });
//     ASSERT_NE(snd_it, h_n_search.table.end());
//     auto thr_it = std::find_if_not(std::next(snd_it), h_n_search.table.end(),
//         [](auto h){ return h == NSearch::EMPTY_HASH; });
//     ASSERT_NE(thr_it, h_n_search.table.end());
//
//     // No other cells
//     EXPECT_EQ(std::find_if_not(std::next(thr_it), h_n_search.table.end(),
//         [](auto h){ return h == NSearch::EMPTY_HASH; }), h_n_search.table.end());
//
//     NSearch fd_n_search = fake_dev_n_search(h_n_search);
//
//     std::array<unsigned, 27 * NSearch::MAX_PARTICLES_IN_CELL> neighbors{};
//
//     unsigned nbr_len = fd_n_search.list_neighbors(glm::vec3{0.1f}, neighbors.data());
//     EXPECT_EQ(nbr_len, 60);
//     for (unsigned i = 0; i < nbr_len; ++i) {
//         unsigned idx = neighbors[i];
//         EXPECT_NE(idx, NSearch::EMPTY_IDX);
//         EXPECT_LT(idx, 60);
//     }
//
//     nbr_len = fd_n_search.list_neighbors(glm::vec3{1.1f}, neighbors.data());
//     EXPECT_EQ(nbr_len, particles);
//     for (unsigned i = 0; i < nbr_len; ++i) {
//         unsigned idx = neighbors[i];
//         EXPECT_NE(idx, NSearch::EMPTY_IDX);
//         EXPECT_LT(idx, particles);
//     }
//
//     nbr_len = fd_n_search.list_neighbors(glm::vec3{2.1f}, neighbors.data());
//     EXPECT_EQ(nbr_len, 60);
//     for (unsigned i = 0; i < nbr_len; ++i) {
//         unsigned idx = neighbors[i];
//         EXPECT_NE(idx, NSearch::EMPTY_IDX);
//         EXPECT_GE(idx, 30);
//         EXPECT_LT(idx, particles);
//     }
// }

int main(int argc, char **argv) {
    cuda_init();

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
