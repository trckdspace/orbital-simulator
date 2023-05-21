#define EIGEN_NO_CUDA

#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/gldraw.h>

#include <iostream>
#include <random>
#include <chrono>
#include <thread>

#include <Eigen/Eigen>

// #include <Simulator.hpp>
#include <MemoryBlock.hpp>
#include <SimulationStep.cuh>

#include <Timer.hpp>
// #include <opencv2/opencv.hpp>

void every(int interval_milliseconds, const std::function<void(void)> &f)
{
    std::thread([f, interval_milliseconds]()
                { 
    while (true)
    {
        auto x = std::chrono::steady_clock::now() + std::chrono::milliseconds(interval_milliseconds);
        f();
        std::this_thread::sleep_until(x);
    } })
        .detach();
}

struct SatelliteSimulator_GPU
{
    const float radius_of_earth_m = 6378100;

    SatelliteSimulator_GPU(size_t width, size_t height) : width(width),
                                                          height(height),
                                                          satellites(width, height),
                                                          positions(width, height)
    {
        std::srand(time(0));
        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<float> dist(0, 1);

        Orbit_f *sats = satellites.host_ptr;
        for (uint64_t i = 0; i < width * height; i++)
        {
            Eigen::Vector3d u = Eigen::Vector3d::Random();

            Eigen::Vector3d v = u.cross(Eigen::Vector3d::Random());
            // Eigen::Vector3d v = Eigen::Vector3d::Random();

            u.normalize();
            v.normalize();

            float min_distance_meters = 100 * 1000;
            float range_in_meters = 5000 * 1000;

            double scaleU = dist(e2) * range_in_meters + (radius_of_earth_m + min_distance_meters);
            // double scaleV = dist(e2) * range_in_meters + (radius_of_earth_m + min_distance_meters);

            u = u.array() * scaleU;
            v = v.array() * scaleU;

            sats[i].u0 = u(0);
            sats[i].u1 = u(1);
            sats[i].u2 = u(2);

            sats[i].v0 = v(0);
            sats[i].v1 = v(1);
            sats[i].v2 = v(2);

            sats[i].t = dist(e2) * M_PI;

            // velocity updated everytime distance is updated.
        }

        if (!satellites.copy_to_device())
        {
            std::cerr << "Couldn't copy" << std::endl;
        }

        positions.setZero_device();

        colors = Eigen::MatrixXf::Random(3, width * height);

        // ? std::cin.get();
    }

    void stepOne(float delta_t)
    {
        dim3 threadsPerBlock(32, 32);
        dim3 numBlocks(width / threadsPerBlock.x + 1, height / threadsPerBlock.y + 1);
        step<<<numBlocks, threadsPerBlock>>>(satellites.device_ptr, positions.device_ptr, width, height, delta_t);
        // cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();
        positions.copy_to_host();
    }

    void draw()
    {
        double scale = radius_of_earth_m;
        glPointSize(20);
        glBegin(GL_POINTS);

        for (int i = 0; i < width * height; i++)
        {
            const auto &pos = positions.host_ptr[i];
            const Eigen::Matrix<float, 3, 1> &color = colors.col(i);

            glColor3f(color(0), color(1), color(2));
            glVertex3f(pos.x / scale, pos.y / scale, pos.z / scale);
        }
        glEnd();
    }

    size_t width, height;
    MemoryBlock<Orbit_f> satellites;
    MemoryBlock<Result> positions;

    Eigen::Matrix<float, 3, -1> colors;
};

int main(int argc, char **argv)
{

    size_t width = atoi(argv[1]);
    size_t height = atoi(argv[2]);

    SatelliteSimulator_GPU sim(width, height);

    std::cerr << "Done" << std::endl;

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
            sim.stepOne(1);
            stepTimer.toc();
            stepTimer.print(); });

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
