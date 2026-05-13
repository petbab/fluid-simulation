#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include "particle_data.cuh"


/**
 * @brief Binary snapshot of fluid particle state (positions + velocities).
 *
 * Boundary particles are NOT stored — they are regenerated from the scene.
 *
 * File layout:
 *   - Header (76 B)
 *   - float4 positions [fluid_n]
 *   - float4 velocities[fluid_n]
 */
class Snapshot {
public:
    static constexpr char     MAGIC[4]     = {'S', 'P', 'H', 'S'};  ///< File magic bytes.
    static constexpr uint32_t VERSION      = 1;                     ///< Snapshot format version.
    static constexpr size_t   APP_NAME_MAX = 64;                    ///< Max application name length.

    /** @brief Snapshot file header. */
    struct Header {
        char     magic[4];        ///< Magic bytes.
        uint32_t version;         ///< Format version.
        char     app_name[APP_NAME_MAX]; ///< Null-terminated scene identifier.
        uint32_t fluid_n;         ///< Number of fluid particles.
    };

    /**
     * @brief Saves fluid particle state to a binary file.
     *
     * Saves first fluid_n positions (from src GL buffer) and velocities to `path`.
     * @param path Destination file path.
     * @param app_name Scene identifier (must fit in APP_NAME_MAX).
     * @param particle_data Particle data containing positions and velocities.
     * @param fluid_n Number of fluid particles to save.
     * @return Error string on app_name overflow or I/O failure; empty on success.
     */
    static std::string save(const std::filesystem::path& path,
                            const std::string& app_name,
                            const ParticleData& particle_data,
                            unsigned fluid_n);

    /**
     * @brief Loads fluid particle state from a binary file.
     *
     * Validates header against `app_name` and `fluid_n`.
     * Writes positions to BOTH src and dst GL position buffers (so a swap
     * doesn't expose stale data) and velocities to the src velocity buffer.
     * Boundary range (offset fluid_n onwards) is left untouched.
     * Derived fields (density, pressure, accels, normal) are not modified —
     * they get overwritten by the next simulation step before being read.
     * @param path Source file path.
     * @param app_name Expected scene identifier.
     * @param particle_data Particle data to load into.
     * @param fluid_n Expected number of fluid particles.
     * @return Error string on header mismatch or I/O failure; empty on success.
     */
    static std::string load(const std::filesystem::path& path,
                            const std::string& app_name,
                            ParticleData& particle_data,
                            unsigned fluid_n);
};

static_assert(sizeof(Snapshot::Header) == 76, "Snapshot::Header layout changed");
