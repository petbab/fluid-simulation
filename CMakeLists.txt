################################################################################
##                            CMake Configuration                             ##
################################################################################

cmake_minimum_required(VERSION 3.22)

# Check if local CMake config exists
if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/CMakeUserConfig.cmake")
    include("${CMAKE_SOURCE_DIR}/cmake/CMakeUserConfig.cmake")
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
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -std=c++20 --extended-lambda")

# Needed for CMake < 3.25
if(NOT DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION)
    set(CMAKE_CUDA20_STANDARD_COMPILE_OPTION "")
    set(CMAKE_CUDA20_EXTENSION_COMPILE_OPTION "")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif()

################################################################################
##                           External Dependencies                            ##
################################################################################

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Open3D REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(nlohmann_json 3.12.0 REQUIRED)

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

include(cmake/ConfigureTarget.cmake)
include(cmake/RegisterApp.cmake)

file(GLOB_RECURSE SOURCE src/*.cpp)
file(GLOB_RECURSE HEADERS src/*.h)
file(GLOB_RECURSE CUDA_SOURCE src/*.cu)
file(GLOB_RECURSE CUDA_HEADERS src/*.cuh)

list(FILTER CUDA_SOURCE EXCLUDE REGEX "src/cuda/tuning/kernels/.*\\.cu$")
list(FILTER CUDA_HEADERS EXCLUDE REGEX "src/cuda/tuning/kernels/.*\\.cuh$")

add_library(fluid_simulation_lib STATIC ${SOURCE} ${HEADERS} ${CUDA_SOURCE} ${CUDA_HEADERS})
configure_target(fluid_simulation_lib)

file(GLOB APP_DIRS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/app/*/")
foreach(APP_DIR ${APP_DIRS})
    add_subdirectory(${APP_DIR})
endforeach()

file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/results")
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/snapshots")
