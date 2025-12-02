################################################################################
##                            CMake Configuration                             ##
################################################################################

cmake_minimum_required(VERSION 3.22)

# Check if local CMake config exists
if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/CMakeUserConfig.cmake")
    include(cmake/CMakeUserConfig.cmake)
else()
    message(FATAL_ERROR
            "CMake configuration not found!\n"
            "Please copy cmake/CMakeUserConfig.cmake.in to cmake/CMakeUserConfig.cmake"
            " and configure it for your system."
    )
endif()
set(CMAKE_INSTALL_PREFIX "~/.local/bin")
set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

project(fluid_simulation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -std=c++20")

# Needed for CMake < 3.25
if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
    set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
    set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
endif()

################################################################################
##                           External Dependencies                            ##
################################################################################

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

find_library(KTT_LIBRARY
        NAMES ktt libktt
        PATHS "${KTT_DIR}/Build/${CMAKE_SYSTEM_PROCESSOR}_${CMAKE_BUILD_TYPE}"
        REQUIRED
)
find_path(KTT_INCLUDE_DIR
        NAMES Ktt.h
        PATHS "${KTT_DIR}/Source" "${KTT_DIR}/include"
        REQUIRED
)
message(STATUS "KTT Library: ${KTT_LIBRARY}")
message(STATUS "KTT Include Directory: ${KTT_INCLUDE_DIR}")

################################################################################
##                            Target Configuration                            ##
################################################################################

file(GLOB_RECURSE SOURCE src/*.cpp)
file(GLOB_RECURSE HEADERS src/*.h)
file(GLOB_RECURSE CUDA_SOURCE src/*.cu)
file(GLOB_RECURSE CUDA_HEADERS src/*.cuh)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCE} ${HEADERS} ${CUDA_SOURCE} ${CUDA_HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
        OpenGL::GL
        glfw
        glad::glad
        OpenMP::OpenMP_CXX
        CompactNSearch
        CUDA::cudart
        CUDA::cuda_driver
        ${KTT_LIBRARY}
)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${KTT_INCLUDE_DIR}
)

target_compile_options(${PROJECT_NAME} PRIVATE  $<$<CONFIG:Release>:-O3>)
target_compile_definitions(${PROJECT_NAME} PRIVATE
        ROOT_DIR="${CMAKE_SOURCE_DIR}"
        $<$<CONFIG:Debug>:DEBUG>
        USE_DOUBLE_PRECISION=OFF
        COMPACT_NSEARCH_STATIC_LIB
        GLM_ENABLE_EXPERIMENTAL # Enable #include glm/gtx/...
        $<$<COMPILE_LANGUAGE:CUDA>:GLM_FORCE_CUDA>
        $<$<COMPILE_LANGUAGE:CUDA>:CUDA_VERSION=${CUDA_VERSION}>
)

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

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


################################################################################
##                                  Testing                                   ##
################################################################################

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
