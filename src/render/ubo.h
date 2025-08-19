#pragma once

#include <glad/glad.h>
#include "../debug.h"


template<class T>
class UBO {
public:
    static constexpr unsigned CAMERA_UBO_BINDING = 0;

    using data_t = T;

    explicit UBO(unsigned binding) {
        glCreateBuffers(1, &ubo);
        glNamedBufferStorage(ubo, sizeof(data_t), nullptr, GL_DYNAMIC_STORAGE_BIT);
        glBindBufferBase(GL_UNIFORM_BUFFER, binding, ubo);
        glCheckError();
    }

    ~UBO() {
        glDeleteBuffers(1, &ubo);
        glCheckError();
    }

    /**
     * Upload data into the UBO on the GPU
     * @param data pointer to the start of the data to be stored (base + offset == data)
     * @param offset offset in the UBO where the data should be stored (not offset in the data)
     * @param size amount of data to write in bytes (base + offset + size <= sizeof(data_t))
     */
    void upload_data(const void *data, unsigned offset = 0, unsigned size = sizeof(data_t)) {
        glNamedBufferSubData(ubo, offset, size, data);
        glCheckError();
    }

private:
    unsigned ubo = 0;
};
