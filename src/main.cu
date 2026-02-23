#include <glfw.h>
#include <app/tilting_box.h>
#include <window.h>
#include <cuda/init.h>
#include <render/asset_manager.h>
#include <exception>


int main() {
#ifndef DEBUG
    try {
#endif

    // Initialize GLFW
    GLFW glfw{};

    Window window{800, 600, "Fluid Simulation"};

    cuda_init();

    {
        Application application{window.get(), window.width(), window.height()};
        application.init();
        application.run();

        // Free the entire application before terminating glfw
        AssetManager::free();
    }

#ifndef DEBUG
    } catch (const std::exception &e) {
        std::cerr << "Exception thrown: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
#endif

    return EXIT_SUCCESS;
}
