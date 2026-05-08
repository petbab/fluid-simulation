#include "snapshot.cuh"

#include <cstring>
#include <fstream>
#include <vector>
#include <thrust/copy.h>
#include <debug.h>


std::string Snapshot::save(const std::filesystem::path& path,
                    const std::string& app_name,
                    const ParticleData& particle_data,
                    unsigned fluid_n) {
    if (app_name.size() >= APP_NAME_MAX)
        return "Snapshot::save: app_name too long (max "
            + std::to_string(APP_NAME_MAX - 1) + " chars)";

    // Pull positions D->H (fluid range only)
    std::vector<float4> positions(fluid_n);
    {
        auto lock = particle_data.position().lock_src();
        cudaMemcpy(positions.data(), lock.get_ptr(),
                   sizeof(float4) * fluid_n, cudaMemcpyDeviceToHost);
        cudaCheckError();
    }

    // Pull velocities D->H
    std::vector<float4> velocities(fluid_n);
    cudaMemcpy(velocities.data(),
               thrust::raw_pointer_cast(particle_data.velocity_vec().data()),
               sizeof(float4) * fluid_n, cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Build header
    Header h{};
    std::memcpy(h.magic, MAGIC, 4);
    h.version = VERSION;
    h.fluid_n = fluid_n;
    std::memcpy(h.app_name, app_name.data(), app_name.size());
    // Remaining bytes of app_name (and any other field) are zeroed by value-init.

    // Write
    std::ofstream f(path, std::ios::binary);
    if (!f)
        return "Snapshot::save: cannot open " + path.string();
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    f.write(reinterpret_cast<const char*>(positions.data()),  sizeof(float4) * fluid_n);
    f.write(reinterpret_cast<const char*>(velocities.data()), sizeof(float4) * fluid_n);
    if (!f)
        return "Snapshot::save: write failed for " + path.string();
    return {};
}

std::string Snapshot::load(const std::filesystem::path& path,
                    const std::string& app_name,
                    ParticleData& particle_data,
                    unsigned fluid_n) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return "Snapshot::load: cannot open " + path.string();

    Header h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f)
        return "Snapshot::load: header read failed for " + path.string();

    if (std::memcmp(h.magic, MAGIC, 4) != 0)
        return "Snapshot::load: bad magic in " + path.string();
    if (h.version != VERSION)
        return "Snapshot::load: version mismatch (file="
            + std::to_string(h.version) + ", expected=" + std::to_string(VERSION) + ")";

    h.app_name[APP_NAME_MAX - 1] = '\0'; // guard against corrupt files
    if (app_name != h.app_name)
        return "Snapshot::load: app_name mismatch (file='"
            + std::string(h.app_name) + "', expected='" + app_name + "')";
    if (h.fluid_n != fluid_n)
        return "Snapshot::load: fluid count mismatch (file="
            + std::to_string(h.fluid_n) + ", expected=" + std::to_string(fluid_n) + ")";

    // Read body
    std::vector<float4> positions(fluid_n);
    std::vector<float4> velocities(fluid_n);
    f.read(reinterpret_cast<char*>(positions.data()),  sizeof(float4) * fluid_n);
    f.read(reinterpret_cast<char*>(velocities.data()), sizeof(float4) * fluid_n);
    if (!f)
        return "Snapshot::load: body read failed for " + path.string();

    // Upload positions to BOTH src and dst (so the next swap doesn't expose stale data).
    // Boundary positions at offset fluid_n are untouched.
    {
        auto lock_src = particle_data.position().lock_src();
        auto lock_dst = particle_data.position().lock_dst();
        cudaMemcpy(lock_src.get_ptr(), positions.data(),
                   sizeof(float4) * fluid_n, cudaMemcpyHostToDevice);
        cudaCheckError();
        cudaMemcpy(lock_dst.get_ptr(), positions.data(),
                   sizeof(float4) * fluid_n, cudaMemcpyHostToDevice);
        cudaCheckError();
    }

    // Upload velocities to src; dst is overwritten by sort+kernel each step.
    thrust::copy(velocities.begin(), velocities.end(),
                 particle_data.velocity_vec().begin());
    return {};
}
