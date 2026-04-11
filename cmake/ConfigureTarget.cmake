function(configure_target TARGET)
    # Link libraries
    target_link_libraries(${TARGET} PUBLIC
            OpenGL::GL
            glfw
            glad::glad
            OpenMP::OpenMP_CXX
            CompactNSearch
            CUDA::cudart
            CUDA::cuda_driver
            ${KTT_LIBRARY}
            Open3D::Open3D
    )
    target_include_directories(${TARGET} PUBLIC
            ${KTT_INCLUDE_DIR}
            "${CMAKE_SOURCE_DIR}/src"
    )

    target_compile_options(${TARGET} PUBLIC  $<$<CONFIG:Release>:-O3>)
    target_compile_definitions(${TARGET} PUBLIC
            ROOT_DIR="${CMAKE_SOURCE_DIR}"
            $<$<CONFIG:Debug>:DEBUG>
            USE_DOUBLE_PRECISION=OFF
            COMPACT_NSEARCH_STATIC_LIB
            GLM_ENABLE_EXPERIMENTAL # Enable #include glm/gtx/...
            $<$<COMPILE_LANGUAGE:CUDA>:GLM_FORCE_CUDA>
            $<$<COMPILE_LANGUAGE:CUDA>:CUDA_VERSION=${CUDA_VERSION}>
            NOT_IN_KTT
    )

    set_target_properties(${TARGET} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
            CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    # Platform-specific settings
    if(WIN32)
        # Windows-specific settings
        set_property(TARGET ${TARGET} PROPERTY WIN32_EXECUTABLE TRUE)

        # On Windows if BUILD_SHARED_LIBS is enabled, copy .dll files to the executable directory
        get_target_property(open3d_type Open3D::Open3D TYPE)
        if(open3d_type STREQUAL "SHARED_LIBRARY")
            set(copy_dlls "${CMAKE_INSTALL_PREFIX}/bin/tbb12$<$<CONFIG:Debug>:_debug>.dll"
                    "${CMAKE_INSTALL_PREFIX}/bin/Open3D.dll")
        else()
            set(copy_dlls "${CMAKE_INSTALL_PREFIX}/bin/tbb12$<$<CONFIG:Debug>:_debug>.dll")
        endif()
        # For CUDA builds, we must also copy the CUDA DLLs. Alternately, add
        # $Env:CUDA_PATH/bin to $Env:PATH to avoid this copy.
        add_custom_command(TARGET ${TARGET} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy ${copy_dlls}
                ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>
                COMMENT "Copying Open3D DLLs to ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
    elseif(APPLE)
        # macOS-specific settings
        find_library(COCOA_LIBRARY Cocoa)
        find_library(IOKIT_LIBRARY IOKit)
        find_library(COREVIDEO_LIBRARY CoreVideo)
        target_link_libraries(${TARGET}
                ${COCOA_LIBRARY}
                ${IOKIT_LIBRARY}
                ${COREVIDEO_LIBRARY}
        )
    endif()
endfunction()
