#ifdef NOT_IN_KTT
#include "common.cuh"
#endif

#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


__global__ void rebuild_n_search(NSearch *dev_n_search, const float4 *particle_positions, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (i == 0) {
        dev_n_search->cell_size = CELL_SIZE;
        dev_n_search->table_size = TABLE_SIZE;
    }

    float4 pos = particle_positions[i];
    NSearch::cell_t c = NSearch::cell_coord(pos);
    NSearch::hash_t pk = NSearch::pack(c);

    if (i == n - 1) {
        dev_n_search->set_cell_end(c, n);
        return;
    }

    float4 next_pos = particle_positions[i + 1];
    NSearch::cell_t next_c = NSearch::cell_coord(next_pos);
    NSearch::hash_t next_pk = NSearch::pack(next_c);

    if (pk != next_pk) {
        dev_n_search->set_cell_end(c, i + 1);
        dev_n_search->set_cell_start(next_c, i + 1);
    }
}
