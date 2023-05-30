#include <pangolin/var/var.h>
#include <pangolin/var/varextra.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/display/widgets.h>
#include <pangolin/display/default_font.h>
#include <pangolin/handler/handler.h>

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

    // while (1)
    // {
    //     stepTimer.tic();
    //     sim.propagate(0.1);
    //     stepTimer.toc();
    //     stepTimer.print();
    // }

    every(10, [&]()
          {
        stepTimer.tic();
        sim.propagate(0.1);
        stepTimer.toc();
        stepTimer.print(); });

    const int UI_WIDTH = 20 * pangolin::default_font().MaxWidth();
    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    pangolin::Var<int> point_size("ui.Point_size", 1, 1, 10);
    pangolin::Var<bool> draw_lines("ui.Draw_lines", false, true);
    pangolin::Var<int> speed("ui.speed", 1, 1, 100);
    pangolin::Var<int> num_satellites("ui.count", 1, 1, atoi(argv[1]));

    while (!pangolin::ShouldQuit())
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        sim.draw_lines = draw_lines;
        sim.speed = speed;

                sim.draw(point_size, num_satellites);
        //  Render OpenGL Cube
        //  pangolin::glDrawColouredCube();
        glColor3f(0.5, 0.5, 0.5);
        pangolin::glDrawAxis(1.0);
        pangolin::glDrawCircle(0, 0, 1);

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }

    return 0;
}
