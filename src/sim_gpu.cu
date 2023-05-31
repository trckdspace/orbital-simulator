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
#include <SimulatorBase.hpp>
#include <Viewer.hpp>

// #include <opencv2/opencv.hpp>

struct SatelliteSimulator_GPU : BaseSimulator
{
    const double radius_of_earth_km = 6378.1;
    const double MEU = 3.986004418e5;

    SatelliteSimulator_GPU(size_t width, size_t height) : BaseSimulator(width * height),
                                                          width(width),
                                                          height(height),
                                                          satellites(width, height),
                                                          positions(width, height)
    {
        std::srand(time(0));
        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<float> dist(0, 1);

        Orbit *sats = satellites.host_ptr;
        Position *state = positions.host_ptr;

        for (uint64_t i = 0; i < width * height; i++)
        {
            Eigen::Vector3d u = Eigen::Vector3d::Random();
            Eigen::Vector3d v = u.cross(Eigen::Vector3d::Random());

            float e = std::abs(u(0));
            // HEO have eccentricy of 0.74 See: https: // en.wikipedia.org/wiki/Molniya_orbit
            e = std::min(e, 0.74f);

            u.normalize();
            v.normalize();

            float perigee = (float(rand() % 500) + (radius_of_earth_km + 160.f));
            float a = perigee / (1 - e);        // std::sqrt((b * b) / (1 - (e * e)));
            float b = a * std::sqrt(1 - e * e); // std::sqrt((b * b) / (1 - (e * e)));
            float f = std::sqrt(a * a - b * b);

            Eigen::Vector3d focus = f * u;

            u *= a;
            v *= b;

            sats[i].rate_multiplier = b / std::sqrt(a / MEU);

            sats[i].u0 = u(0);
            sats[i].u1 = u(1);
            sats[i].u2 = u(2);

            sats[i].v0 = v(0);
            sats[i].v1 = v(1);
            sats[i].v2 = v(2);

            sats[i].c0 = focus(0);
            sats[i].c1 = focus(1);
            sats[i].c2 = focus(2);

            sats[i].t = u(0) * M_PI * 2; // rand() * 2 * M_PI;

            sats[i].propagate(state[i], 0, true);

            // velocity updated everytime distance is updated.
        }

        if (!satellites.copy_to_device())
        {
            std::cerr << "Couldn't copy" << std::endl;
        }

        if (!positions.copy_to_device())
        {
            std::cerr << "Couldn't copy positions" << std::endl;
        }

        colors = Eigen::MatrixXf::Random(3, width * height);

        // ? std::cin.get();
    }

    virtual void propagate(double delta_t)
    {
        dim3 threadsPerBlock(32, 32);
        dim3 numBlocks(width / threadsPerBlock.x + 1, height / threadsPerBlock.y + 1);
        step<<<numBlocks, threadsPerBlock>>>(satellites.device_ptr, positions.device_ptr, width, height, speed * delta_t);
        cudaDeviceSynchronize();
        CHECK_LAST_CUDA_ERROR();
    }

    virtual void draw(int point_size, int num_satellites)
    {
        double scale = radius_of_earth_km;
        glPointSize(point_size);
        glBegin(GL_POINTS);

        for (int i = 0; i < std::min(num_satellites,int(width * height)); i++)
        {
            const auto &pos = positions.host_ptr[i];
            const Eigen::Matrix<float, 3, 1> &color = colors.col(i);

            glColor3f(color(0), color(1), color(2));
            glVertex3f(pos.x / scale, pos.y / scale, pos.z / scale);
        }
        glEnd();
    }

    size_t width, height;
    MemoryBlock<Orbit> satellites;
    MemoryBlock<Position> positions;

    Eigen::Matrix<float, 3, -1> colors;
};

int main(int argc, char **argv)
{

    size_t width = atoi(argv[1]);
    size_t height = atoi(argv[2]);

    SatelliteSimulator_GPU sim(width, height);
    Viewer viewer(&sim);

    viewer.mainLoop();
    return 0;
}
