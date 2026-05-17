#include "config_loader.h"
#include <fstream>
#include <stdexcept>
#define NLOHMANN_JSON_NO_STD_RANGES
#include <nlohmann/json.hpp>


ktt::KernelConfiguration load_config_json(const std::filesystem::path& path,
                                          ktt::Tuner& tuner,
                                          ktt::KernelId kernel) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot open config file: " + path.string());

    nlohmann::json j;
    f >> j;

    nlohmann::json::array_t arr;
    if (j.is_array()) {
        arr = j.get<nlohmann::json::array_t>();
    } else if (j.contains("Configuration") && j["Configuration"].is_array()) {
        arr = j["Configuration"].get<nlohmann::json::array_t>();
    } else {
        throw std::runtime_error("Config JSON must be an array or contain a 'Configuration' array");
    }

    ktt::ParameterInput input;
    for (const auto& entry : arr) {
        if (!entry.contains("Name") || !entry.contains("Value") || !entry.contains("ValueType"))
            throw std::runtime_error("Config entry missing Name, Value, or ValueType");

        std::string name = entry["Name"];
        std::string value_type = entry["ValueType"];
        ktt::ParameterValue value;

        if (value_type == "Double") {
            value = entry["Value"].get<double>();
        } else if (value_type == "UnsignedInt") {
            value = entry["Value"].get<uint64_t>();
        } else if (value_type == "Int") {
            value = entry["Value"].get<int64_t>();
        } else if (value_type == "Bool") {
            value = entry["Value"].get<bool>();
        } else if (value_type == "String") {
            value = entry["Value"].get<std::string>();
        } else {
            throw std::runtime_error("Unknown ValueType: " + value_type);
        }

        input.emplace_back(name, value);
    }

    return tuner.CreateConfiguration(kernel, input);
}
