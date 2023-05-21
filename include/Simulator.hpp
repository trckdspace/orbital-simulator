#pragma once

#define EIGEN_NO_CUDA

#include <Eigen/Eigen>
#include <GL/gl.h>

#include <chrono>
#include <thread>
#include <iostream>

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

template <int DIM = 3>
struct Simulation
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    const float radius_of_earth_km = 6378.1;

    typedef Eigen::MatrixXf MatrixType;

    Simulation(int orbit_count = 100) : numberOfOrbits(orbit_count)
    {
        U = MatrixType::Random(DIM, numberOfOrbits);
        V = MatrixType::Random(DIM, numberOfOrbits);
        t = MatrixType::Random(1, numberOfOrbits);
        state = MatrixType::Zero(DIM, numberOfOrbits);

        for (int i = 0; i < numberOfOrbits; i++)
        {

            Eigen::Matrix<float, 3, 1> u = U.col(i).topRows(3);
            Eigen::Matrix<float, 3, 1> v = V.col(i).topRows(3);

            u.normalize();
            v.normalize();

            // v(0) = 1;
            // v(1) = 0;
            // v(2) = 0;

            // u(0) = 0;
            // u(1) = 0;
            // u(2) = 1;

            // std::cerr << u.norm() << " " << v.norm() << std::endl;

            // std::cerr << u.rows() << "x" << u.cols() << " " << U.col(i).rows() << "x" << U.col(i).cols() << std::endl;

            Eigen::Matrix<float, 3, 1> v_n = u.cross(v);

            float scale = (float(rand() % 500) + (radius_of_earth_km + 300.f));

            u *= scale;
            v_n *= scale;

            // std::cerr << scale << std::endl;

            U.col(i).topRows(3) = u;
            V.col(i).topRows(3) = v_n;

            // std::cerr << U.col(i).transpose() << " " << V.col(i).transpose() << std::endl;
        }
        colors = MatrixType::Random(DIM, numberOfOrbits);
    }

    void propagate(double dt)
    {
        t = t.array() + dt;
        state = Eigen::cos(t.array()).replicate(DIM, 1) * U.array() +
                Eigen::sin(t.array()).replicate(DIM, 1) * V.array();
    }

    void draw()
    {
        double scale = radius_of_earth_km;
        glPointSize(10);
        glBegin(GL_POINTS);

        for (int i = 0; i < numberOfOrbits; i++)
        {
            const Eigen::Matrix<float, 3, 1> &pos = state.col(i);
            const Eigen::Matrix<float, 3, 1> &color = colors.col(i);
            glColor3f(color(0), color(1), color(2));
            glVertex3f(pos(0) / scale, pos(1) / scale, pos(2) / scale);
        }
        glEnd();
    }

    int numberOfOrbits = 100;
    Eigen::Matrix<float, DIM, -1> U, V, state, colors;
    Eigen::Matrix<float, 1, -1> t;
};
