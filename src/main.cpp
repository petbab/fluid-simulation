#include "glfw.h"
#include "application.h"
#include "window.h"


int main() {
    // Initialize GLFW
    GLFW glfw{};

    Window window{800, 600, "Fluid Simulation"};

    {
        Application application{window.get(), window.width(), window.height()};
        application.run();

        // Free the entire application before terminating glfw
    }

    return 0;
}
