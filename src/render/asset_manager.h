#pragma once

#include <memory>
#include <type_traits>
#include "shader.h"
#include "geometry.h"
#include "object.h"


class AssetManager {
public:
    AssetManager(const AssetManager&) = delete;
    AssetManager& operator=(const AssetManager&) = delete;

    static void free() {
        instance().shaders.clear();
        instance().geometries.clear();
        instance().objects.clear();
    }

    template<class T, class... Args>
    requires std::derived_from<T, Shader> ||
             std::derived_from<T, Geometry> ||
             std::derived_from<T, Object>
    static T* make(const std::string &name, Args&&... args) {
        return make_impl<T>(get_container<T>(), name, std::forward<Args>(args)...);
    }

    template<class T>
    requires std::derived_from<T, Shader> ||
             std::derived_from<T, Geometry> ||
             std::derived_from<T, Object>
    static T* get(const std::string &name) {
        return get_impl<T>(get_container<T>(), name);
    }

private:
    AssetManager() = default;

    static AssetManager& instance() {
        static AssetManager m{};
        return m;
    }

    template<class T>
    static auto& get_container() {
        if constexpr (std::derived_from<T, Shader>)
            return instance().shaders;
        if constexpr (std::derived_from<T, Geometry>)
            return instance().geometries;
        if constexpr (std::derived_from<T, Object>)
            return instance().objects;
    }

    template<class T, class Container, class... Args>
    static T* make_impl(Container& container, const std::string &name, Args&&... args) {
        auto [it, inserted] = container.emplace(
            name, std::make_unique<T>(std::forward<Args>(args)...));
        return static_cast<T*>(it->second.get());
    }

    template<class T, class Container>
    static T* get_impl(Container& container, const std::string &name) {
        auto it = container.find(name);
        return it != container.end()
               ? static_cast<T*>(it->second.get())
               : nullptr;
    }

    std::unordered_map<std::string, std::unique_ptr<Shader>> shaders{};
    std::unordered_map<std::string, std::unique_ptr<Geometry>> geometries{};
    std::unordered_map<std::string, std::unique_ptr<Object>> objects{};
};
