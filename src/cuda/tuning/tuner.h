#pragma once

#include <Ktt.h>


/**
 * @brief Base class for KTT kernel auto-tuners.
 *
 * Manages a KTT tuner instance, kernel definitions, and results.
 * Provides common infrastructure for running kernels with or without tuning.
 */
class Tuner {
public:
    explicit Tuner();
    virtual ~Tuner();

    /**
     * @brief Returns tuning statistics.
     * @return Pair of (configurations searched, total configurations).
     */
    std::pair<int, int> tuning_stats() const;

    /** @brief Clears cached configuration data. */
    void clear_configuration_data();

    /**
     * @brief Prints the best found configuration.
     * @param out Output stream.
     */
    void print_best_config(std::ostream& out) const;

protected:
    /**
     * @brief Runs the kernel with optional tuning.
     * @param tune If true, searches for better configurations.
     * @return Kernel result.
     */
    ktt::KernelResult run(bool tune);

    /**
     * @brief Updates the argument list for the next run.
     * @param new_args New argument IDs.
     */
    void update_args(const std::vector<ktt::ArgumentId>& new_args);

private:
    static ktt::Tuner* instance();  ///< Singleton KTT tuner instance.

protected:
    ktt::Tuner* tuner;              ///< KTT tuner handle.

    std::string name;               ///< Tuner name.
    ktt::KernelDefinitionId definition = 0;  ///< Current kernel definition ID.
    ktt::KernelId kernel = 0;       ///< Current kernel ID.
    std::vector<ktt::ArgumentId> args;  ///< Current argument IDs.

    int searched_count = 0;         ///< Number of configurations searched.
    std::vector<ktt::KernelResult> results;  ///< Collected tuning results.
};
