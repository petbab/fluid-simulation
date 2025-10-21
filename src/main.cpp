#include "glfw.h"
#include "application.h"
#include "window.h"
#include "cuda/init.h"
#include "render/asset_manager.h"


int main() {
    // Initialize GLFW
    GLFW glfw{};

    Window window{800, 600, "Fluid Simulation"};

    cuda_init();

    {
        Application application{window.get(), window.width(), window.height()};
        application.run();

        // Free the entire application before terminating glfw
        AssetManager::free();
    }

    return 0;
}
