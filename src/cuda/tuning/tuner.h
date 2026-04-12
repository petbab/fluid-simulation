#pragma once

#include <Ktt.h>


class Tuner {
public:
    Tuner();
    virtual ~Tuner();

protected:
    ktt::KernelResult run(bool tune);

private:
    void print_best_config() const;
    static ktt::Tuner* instance();

protected:
    ktt::Tuner* tuner;
    ktt::KernelDefinitionId definition = 0;
    ktt::KernelId kernel = 0;
    bool has_tuned = false;
};
