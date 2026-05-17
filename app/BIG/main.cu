#include <glfw.h>
#include "application.h"
#include <window.h>
#include <cuda/init.h>
#include <render/asset_manager.h>
#include <exception>


int main(int argc, char** argv) {
#ifndef DEBUG
    try {
#endif

    RunOptions opts = parse_cli(argc, argv);

    // Initialize GLFW
    GLFW glfw{};
    if (opts.headless)
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    Window window{800, 600, APP_NAME};

    cuda_init();

    {
        BIGApp application{window.get(), window.width(), window.height(), APP_NAME, opts};
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
