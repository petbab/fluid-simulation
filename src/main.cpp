#include "glfw.h"
#include "application.h"


int main() {
    // Initialize GLFW
    GLFW glfw{};

    {
        Application application{800, 600, "Fluid Simulation"};
        application.run();

        // Free the entire application before terminating glfw
    }

    return 0;
}
