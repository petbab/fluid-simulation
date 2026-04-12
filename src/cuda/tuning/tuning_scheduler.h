#pragma once

#include <vector>
#include <algorithm>


class TuningScheduler {
public:
    TuningScheduler(int tuners_count, float tune_iterations_per_frame)
        : scheduled_tuners(tuners_count), tune_iterations_per_frame(tune_iterations_per_frame) {}

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

    bool is_scheduled(int tuner_i) const { return scheduled_tuners[tuner_i]; }

    void set_tune_iterations_per_frame(float tipf) {
        tune_iterations_per_frame = tipf;
    }

private:
    std::vector<bool> scheduled_tuners;
    float tune_iterations_per_frame;
    int offset = 0;
    float accumulator = 0.f;
};
