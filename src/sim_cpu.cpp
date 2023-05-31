
#include <Viewer.hpp>
#include <Simulator.hpp>

int main(int argc, char **argv)
{
    srand(time(0));
    Simulation<3> sim(atoi(argv[1]));

    Viewer viewer(&sim);

    viewer.mainLoop();

    return 0;
}
