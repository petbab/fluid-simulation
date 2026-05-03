#pragma once

#include <Ktt.h>


class Tuner {
public:
    Tuner();
    virtual ~Tuner();

    std::pair<int, int> tuning_stats() const;
    void clear_configuration_data();

protected:
    ktt::KernelResult run(bool tune);
    void update_args(const std::vector<ktt::ArgumentId>& new_args);

private:
    void print_best_config() const;
    static ktt::Tuner* instance();

protected:
    ktt::Tuner* tuner;
    ktt::KernelDefinitionId definition = 0;
    ktt::KernelId kernel = 0;
    std::vector<ktt::ArgumentId> args;
    int searched_count = 0;
};
