#include <Viewer.hpp>
#include <SimulatorSGP4.hpp>

int main(int argc, char **argv)
{
    SimulatorSGP4 sim(argv[1]);

    Viewer viewer(&sim);

    viewer.mainLoop();

    return 0;
}