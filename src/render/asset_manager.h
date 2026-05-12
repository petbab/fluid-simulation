#pragma once

#include <memory>
#include <ranges>
#include <type_traits>
#include "shader.h"
#include "geometry.h"
#include "object.h"


/**
 * @brief Singleton manager for render assets (shaders, geometries, objects).
 *
 * Provides type-safe creation and retrieval of Shader, Geometry, and Object
 * instances identified by unique names. All assets are stored in internal
 * containers and managed via std::unique_ptr.
 */
class AssetManager {
    template<class T>
    using container_t = std::unordered_map<std::string, std::unique_ptr<T>>;

public:
    AssetManager(const AssetManager&) = delete;
    AssetManager& operator=(const AssetManager&) = delete;

    /**
     * @brief Clears all stored assets (shaders, geometries, objects).
     */
    static void free() {
        instance().shaders.clear();
        instance().geometries.clear();
        instance().objects.clear();
    }

    /**
     * @brief Creates a new asset of type T with the given name and arguments.
     * @tparam T Asset type (must derive from Shader, Geometry, or Object).
     * @tparam Args Constructor argument types.
     * @param name Unique identifier for the asset.
     * @param args Arguments forwarded to the constructor of T.
     * @return Pointer to the created asset.
     */
    template<class T, class... Args>
    requires std::derived_from<T, Shader> ||
             std::derived_from<T, Geometry> ||
             std::derived_from<T, Object>
    static T* make(const std::string &name, Args&&... args) {
        return make_impl<T>(get_container<T>(), name, std::forward<Args>(args)...);
    }

    /**
     * @brief Retrieves an existing asset by name.
     * @tparam T Asset type (must derive from Shader, Geometry, or Object).
     * @param name Unique identifier of the asset.
     * @return Pointer to the asset, or nullptr if not found.
     */
    template<class T>
    requires std::derived_from<T, Shader> ||
             std::derived_from<T, Geometry> ||
             std::derived_from<T, Object>
    static T* get(const std::string &name) {
        return get_impl<T>(get_container<T>(), name);
    }

    /**
     * @brief Returns a view over all stored assets of type T.
     * @tparam T Asset type (must derive from Shader, Geometry, or Object).
     * @return A range of raw pointers to the stored assets.
     */
    template<class T>
    requires std::derived_from<T, Shader> ||
             std::derived_from<T, Geometry> ||
             std::derived_from<T, Object>
    static auto container() {
        return get_container<T>() | std::ranges::views::transform([](const auto &p){ return p.second.get(); });
    }

private:
    AssetManager() = default;

    /**
     * @brief Returns the singleton instance.
     * @return Reference to the AssetManager instance.
     */
    static AssetManager& instance() {
        static AssetManager m{};
        return m;
    }

    /**
     * @brief Returns the internal container for the given asset type.
     * @tparam T Asset type.
     * @return Reference to the corresponding unordered_map.
     */
    template<class T>
    static auto& get_container() {
        if constexpr (std::derived_from<T, Shader>)
            return instance().shaders;
        if constexpr (std::derived_from<T, Geometry>)
            return instance().geometries;
        if constexpr (std::derived_from<T, Object>)
            return instance().objects;
    }

    /**
     * @brief Implementation of asset creation.
     * @tparam T Asset type.
     * @tparam Container Storage container type.
     * @tparam Args Constructor argument types.
     * @param container The container to store the asset in.
     * @param name Unique identifier for the asset.
     * @param args Arguments forwarded to the constructor of T.
     * @return Pointer to the created asset.
     */
    template<class T, class Container, class... Args>
    static T* make_impl(Container& container, const std::string &name, Args&&... args) {
        auto [it, inserted] = container.emplace(
            name, std::make_unique<T>(std::forward<Args>(args)...));
        return static_cast<T*>(it->second.get());
    }

    /**
     * @brief Implementation of asset retrieval.
     * @tparam T Asset type.
     * @tparam Container Storage container type.
     * @param container The container to search in.
     * @param name Unique identifier of the asset.
     * @return Pointer to the asset, or nullptr if not found.
     */
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
