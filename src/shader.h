#pragma once

#include <filesystem>


class Shader {
public:
    Shader(const char *vertex, const char *fragment);
    Shader(const std::filesystem::path& vertex, const std::filesystem::path& fragment);

    ~Shader();

    void use() const;

private:
    unsigned program;
};
