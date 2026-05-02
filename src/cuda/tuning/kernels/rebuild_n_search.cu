#define KERNEL_PATH(file) <KERNEL_DIR ## file>

#include KERNEL_PATH(/common.cuh)


__global__ void rebuild_n_search(NSearch *dev_n_search, const float4 *particle_positions, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 pos = particle_positions[i];
    NSearch::hash_t h = NSearch::pos_to_cell_hash(pos, dev_n_search->cell_size);

    if (i == n - 1) {
        dev_n_search->set_cell_end(h, n);
        return;
    }

    float4 next_pos = particle_positions[i + 1];
    NSearch::hash_t next_h = NSearch::pos_to_cell_hash(next_pos, dev_n_search->cell_size);

    if (h != next_h) {
        dev_n_search->set_cell_end(h, i + 1);
        dev_n_search->set_cell_start(next_h, i + 1);
    }
}
