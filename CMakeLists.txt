cmake_minimum_required(VERSION 3.22)

# Local settings:
# `nvidia-smi --query-gpu=name,compute_cap --format=csv`
# `which nvcc`
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(fluid_simulation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(OpenMP REQUIRED) # TODO: pull in `libomp` from vcpkg for Clang
find_package(CUDAToolkit REQUIRED)

# CompactNSearch
include(FetchContent)
set(USE_DOUBLE_PRECISION OFF CACHE BOOL "Use double precision")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
FetchContent_Declare(
        CompactNSearch
        GIT_REPOSITORY https://github.com/petbab/CompactNSearch.git
        GIT_TAG master
)
FetchContent_MakeAvailable(CompactNSearch)
# Ensure CompactNSearch is built with the same flags
if(TARGET CompactNSearch)
    target_compile_options(CompactNSearch PRIVATE $<$<CONFIG:Release>:-O3>)
    target_compile_features(CompactNSearch PRIVATE cxx_std_20)
endif()

file(GLOB_RECURSE SOURCE src/*.cpp)
file(GLOB_RECURSE HEADERS src/*.h)
file(GLOB_RECURSE CUDA_SOURCE src/*.cu)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCE} ${HEADERS} ${CUDA_SOURCE})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
        OpenGL::GL
        glfw
        glad::glad
        OpenMP::OpenMP_CXX
        CompactNSearch
        CUDA::cudart
)

target_compile_options(${PROJECT_NAME} PRIVATE  $<$<CONFIG:Release>:-O3>)
target_compile_definitions(${PROJECT_NAME} PRIVATE
        ROOT_DIR="${CMAKE_SOURCE_DIR}"
        $<$<CONFIG:Debug>:DEBUG>
        USE_DOUBLE_PRECISION=OFF
        COMPACT_NSEARCH_STATIC_LIB
        GLM_ENABLE_EXPERIMENTAL # Enable #include glm/gtx/...
)

# Platform-specific settings
if(WIN32)
    # Windows-specific settings
    set_property(TARGET ${PROJECT_NAME} PROPERTY WIN32_EXECUTABLE TRUE)
elseif(APPLE)
    # macOS-specific settings
    find_library(COCOA_LIBRARY Cocoa)
    find_library(IOKIT_LIBRARY IOKit)
    find_library(COREVIDEO_LIBRARY CoreVideo)
    target_link_libraries(${PROJECT_NAME}
            ${COCOA_LIBRARY}
            ${IOKIT_LIBRARY}
            ${COREVIDEO_LIBRARY}
    )
endif()

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# GTest
find_package(GTest CONFIG REQUIRED)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

enable_testing()

file(GLOB TESTS test/*.cpp)
add_executable("${PROJECT_NAME}_test" ${TESTS})
target_link_libraries(
        "${PROJECT_NAME}_test"
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests("${PROJECT_NAME}_test")
