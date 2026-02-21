
## Prerequisites
```
sudo apt install libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev pkg-config
```
[CUDA toolkit](https://developer.nvidia.com/cuda-toolkit): runtime and headers

[vcpkg setup](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started?pivots=shell-bash)

[Open3D](https://github.com/isl-org/Open3D/tree/main/examples/cmake/open3d-cmake-find-package)

## CMake Setup
1. Copy `cmake/CMakeUserConfig.cmake.in` to `cmake/CMakeUserConfig.cmake` and configure it for your system.
1. Build the project:
    ```
    mkdir build
    cd build
    cmake ..
    cmake --build . -j 10
    ```
