vcpkg setup: https://learn.microsoft.com/en-us/vcpkg/get_started/get-started?pivots=shell-bash

## Prerequisites
```
sudo apt install libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev pkg-config
```

## CMake Setup
```
cmake --preset=vcpkg -B build/
cd build/
make install
```
