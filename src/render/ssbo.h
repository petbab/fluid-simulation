#pragma once

#include <glad/glad.h>
#include <debug.h>


/**
 * @brief Shader Storage Buffer Object (SSBO) wrapper.
 *
 * Provides GPU-backed storage for arbitrary data types with dynamic updates.
 * Typically used for passing large arrays to compute or vertex shaders.
 * @tparam T Element type stored in the buffer.
 */
template<typename T>
class SSBO {
public:
    static constexpr unsigned VISUALIZE_VEC3_SSBO_BINDING = 0;   ///< Binding for vec3 visualization data.
    static constexpr unsigned VISUALIZE_FLOAT_SSBO_BINDING = 1;  ///< Binding for float visualization data.
    static constexpr unsigned VISUALIZE_UINT_SSBO_BINDING = 2;   ///< Binding for uint visualization data.

    /**
     * @brief Constructs an SSBO with space for count elements.
     * @param count Number of T elements to allocate.
     */
    SSBO(unsigned count) : count{count} {
        glCreateBuffers(1, &ssbo);
        glNamedBufferStorage(ssbo, sizeof(T) * count, nullptr, GL_DYNAMIC_STORAGE_BIT);
        glCheckError();
    }

    ~SSBO() {
        glDeleteBuffers(1, &ssbo);
        glCheckError();
    }

    /**
     * @brief Uploads a subrange of data into the SSBO.
     * @param data Pointer to the source data.
     * @param size Number of bytes to write.
     * @param offset Byte offset in the SSBO where writing starts.
     */
    void upload_data(const void *data, unsigned size, unsigned offset = 0) {
        glNamedBufferSubData(ssbo, offset, size, data);
        glCheckError();
    }

    /**
     * @brief Uploads the entire buffer contents.
     * @param data Pointer to the source data (must hold count elements).
     */
    void upload_data(const void *data) {
        upload_data(data, sizeof(T) * count);
    }

    /**
     * @brief Binds the SSBO to a shader storage buffer binding point.
     * @param binding Binding index.
     */
    void bind(unsigned binding) const {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, ssbo);
        glCheckError();
    }

    /** @return The OpenGL SSBO identifier. */
    unsigned get_id() const { return ssbo; }

private:
    unsigned ssbo = 0;  ///< OpenGL buffer identifier.
    unsigned count;     ///< Number of elements allocated.
};
