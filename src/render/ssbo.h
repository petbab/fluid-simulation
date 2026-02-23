#pragma once

#include <glad/glad.h>
#include <debug.h>


template<typename T>
class SSBO {
public:
    static constexpr unsigned VISUALIZE_VEC3_SSBO_BINDING = 0;
    static constexpr unsigned VISUALIZE_FLOAT_SSBO_BINDING = 1;

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
     * Upload data into the SSBO on the GPU
     * @param data pointer to the start of the data to be stored (base + offset == data)
     * @param size amount of data to write in bytes (base + offset + size <= sizeof(data_t))
     * @param offset offset in the SSBO where the data should be stored (not offset in the data)
     */
    void upload_data(const void *data, unsigned size, unsigned offset = 0) {
        glNamedBufferSubData(ssbo, offset, size, data);
        glCheckError();
    }

    void upload_data(const void *data) {
        upload_data(data, sizeof(T) * count);
    }

    void bind(unsigned binding) const {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, ssbo);
        glCheckError();
    }

    unsigned get_id() const { return ssbo; }

private:
    unsigned ssbo = 0;
    unsigned count;
};
