#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/gldraw.h>

#include <Simulator.hpp>
#include <Timer.hpp>

int main(int argc, char **argv)
{
    srand(time(0));

    Simulation<3> sim(atoi(argv[1]));

    pangolin::CreateWindowAndBind("Main", 640, 480);
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
        pangolin::ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin::AxisY));

    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                                .SetHandler(&handler);

    Timer stepTimer("stepTimer");
    every(10, [&]()
          {
              stepTimer.tic();
              sim.propagate(1);
              stepTimer.toc();
              // stepTimer.print();
          });

    while (!pangolin::ShouldQuit())
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        sim.draw();
        // Render OpenGL Cube
        // pangolin::glDrawColouredCube();
        pangolin::glDrawAxis(1.0);
        pangolin::glDrawCircle(0, 0, 1);

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }

    return 0;
}
