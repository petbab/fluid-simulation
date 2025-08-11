cmake_minimum_required(VERSION 3.30)
project(fluid_simulation)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)

file(GLOB SOURCE src/*.cpp)
file(GLOB HEADERS src/*.h)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCE} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
        OpenGL::GL
        glfw
        glad::glad
)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Werror $<$<CONFIG:Release>:-O3>)
target_compile_definitions(${PROJECT_NAME} PRIVATE
        ROOT_DIR="${CMAKE_SOURCE_DIR}"
        $<$<CONFIG:Debug>:DEBUG>
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
