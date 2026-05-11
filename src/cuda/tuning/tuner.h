#pragma once

#include <Ktt.h>
#include <cli.h>


class Tuner {
public:
    explicit Tuner();
    virtual ~Tuner();

    std::pair<int, int> tuning_stats() const;
    void clear_configuration_data();
    void print_best_config(std::ostream& out) const;

    void set_searcher(RunOptions::Searcher s);
    void set_results_out(std::optional<std::filesystem::path> out);
    virtual void set_frozen_config(ktt::KernelConfiguration cfg);
    void clear_frozen_config();

    ktt::Tuner* get_tuner() const { return tuner; }
    ktt::KernelId get_kernel() const { return kernel; }

protected:
    ktt::KernelResult run(bool tune);
    void update_args(const std::vector<ktt::ArgumentId>& new_args);

private:
    static ktt::Tuner* instance();

protected:
    ktt::Tuner* tuner;

    std::string name;
    ktt::KernelDefinitionId definition = 0;
    ktt::KernelId kernel = 0;
    std::vector<ktt::ArgumentId> args;

    int searched_count = 0;
    std::vector<ktt::KernelResult> results;

    std::optional<std::filesystem::path> results_out;
    std::optional<ktt::KernelConfiguration> frozen_config;
};
