#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include "particle_data.cuh"


// Binary snapshot of fluid particle state (positions + velocities).
// Boundary particles are NOT stored — they are regenerated from the scene.
//
// File layout:
//   Header (76 B)
//   float4 positions [fluid_n]
//   float4 velocities[fluid_n]
class Snapshot {
public:
    static constexpr char     MAGIC[4]     = {'S', 'P', 'H', 'S'};
    static constexpr uint32_t VERSION      = 1;
    static constexpr size_t   APP_NAME_MAX = 64;

    struct Header {
        char     magic[4];
        uint32_t version;
        char     app_name[APP_NAME_MAX]; // null-terminated; identifies the scene
        uint32_t fluid_n;
    };

    // Saves first fluid_n positions (from src GL buffer) and velocities to `path`.
    // Returns error string on app_name overflow or I/O failure.
    static std::string save(const std::filesystem::path& path,
                            const std::string& app_name,
                            const ParticleData& particle_data,
                            unsigned fluid_n);

    // Loads positions and velocities from `path` into particle_data.
    // Validates header against `app_name` and `fluid_n`.
    // Returns error string on header mismatch or I/O failure.
    //
    // Writes positions to BOTH src and dst GL position buffers (so a swap
    // doesn't expose stale data) and velocities to the src velocity buffer.
    // Boundary range (offset fluid_n onwards) is left untouched.
    // Derived fields (density, pressure, accels, normal) are not modified —
    // they get overwritten by the next simulation step before being read.
    static std::string load(const std::filesystem::path& path,
                            const std::string& app_name,
                            ParticleData& particle_data,
                            unsigned fluid_n);
};

static_assert(sizeof(Snapshot::Header) == 76, "Snapshot::Header layout changed");
