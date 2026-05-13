#pragma once

#include <vector>
#include <algorithm>


/**
 * @brief Round-robin scheduler for distributing tuning budget across tuners.
 *
 * Maintains a fractional accumulator and schedules a subset of tuners
 * each frame based on tune_iterations_per_frame.
 */
class TuningScheduler {
public:
    /**
     * @brief Constructs the scheduler.
     * @param tuners_count Number of tuners to schedule.
     * @param tune_iterations_per_frame Average tuners to run per frame.
     */
    TuningScheduler(int tuners_count, float tune_iterations_per_frame)
        : scheduled_tuners(tuners_count), tune_iterations_per_frame(tune_iterations_per_frame) {}

    /** @brief Advances the schedule for the current frame. */
    void schedule() {
        accumulator += tune_iterations_per_frame;

        const int n = static_cast<int>(scheduled_tuners.size());
        const int budget = std::min(static_cast<int>(accumulator), n);
        accumulator -= budget;

        int i = 0;
        for (; i < budget; ++i)
            scheduled_tuners[(offset + i) % n] = true;
        for (; i < n; ++i)
            scheduled_tuners[(offset + i) % n] = false;

        offset = (offset + budget) % n;
    }

    /**
     * @brief Checks if a tuner is scheduled for this frame.
     * @param tuner_i Tuner index.
     * @return True if scheduled.
     */
    bool is_scheduled(int tuner_i) const { return scheduled_tuners[tuner_i]; }

    /**
     * @brief Sets the tuning iterations per frame.
     * @param tipf New value.
     */
    void set_tune_iterations_per_frame(float tipf) {
        tune_iterations_per_frame = tipf;
        accumulator = 0.f;
    }

private:
    std::vector<bool> scheduled_tuners;  ///< Per-tuner schedule flags.
    float tune_iterations_per_frame;     ///< Target tuners per frame.
    int offset = 0;                      ///< Round-robin offset.
    float accumulator = 0.f;             ///< Fractional accumulator.
};
