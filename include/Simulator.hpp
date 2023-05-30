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
        t = MatrixType::Random(001, numberOfOrbits) * 2 * M_PI;
        C = MatrixType::Zero(DIM, numberOfOrbits);

        state = MatrixType::Zero(DIM, numberOfOrbits);
        rate_multiplier = MatrixType::Zero(1, numberOfOrbits);

        for (int i = 0; i < numberOfOrbits; i++)
        {
            Eigen::Matrix<float, 3, 1> u = U.col(i).topRows(3);
            Eigen::Matrix<float, 3, 1> v = V.col(i).topRows(3);

            float e = std::abs(U(0, i));
            // HEO have eccentricy of 0.74 See: https: // en.wikipedia.org/wiki/Molniya_orbit

            e = std::min(e, 0.74f);

            u.normalize();
            v.normalize();

            Eigen::Matrix<float, 3, 1> v_n = u.cross(v);

            v_n.normalize();

            float perigee = (float(rand() % 2000) + (radius_of_earth_km + 160.f));
            float a = perigee / (1 - e);        // std::sqrt((b * b) / (1 - (e * e)));
            float b = a * std::sqrt(1 - e * e); // std::sqrt((b * b) / (1 - (e * e)));
            float f = std::sqrt(a * a - b * b);

            U.col(i).topRows(3) = a * u;
            V.col(i).topRows(3) = b * v_n;
            C.col(i).topRows(3) = f * u;

            rate_multiplier(i) = b / sqrt(a / MEU);

            // std::cerr << "INFO: " << a << " " << b << std::endl;

            // std::cerr << U.col(i).transpose() << " " << V.col(i).transpose() << std::endl;
        }
        colors = MatrixType::Random(DIM, numberOfOrbits);

        state = C.array() + Eigen::cos(t.array()).replicate(DIM, 1) * U.array() +
                Eigen::sin(t.array()).replicate(DIM, 1) * V.array();
    }

    void propagate(double dt)
    {
        // t = t.array();

        t = t.array() + dt * rate_multiplier.array() / (state).colwise().squaredNorm().array();

        state = C.array() + Eigen::cos(t.array()).replicate(DIM, 1) * U.array() +
                Eigen::sin(t.array()).replicate(DIM, 1) * V.array();
    }

    void draw()
    {
        double scale = radius_of_earth_km;
        glPointSize(10);

        for (int i = 0; i < numberOfOrbits; i++)
        {
            const Eigen::Matrix<float, 3, 1> &pos = state.col(i);
            const Eigen::Matrix<float, 3, 1> &color = colors.col(i);
            const Eigen::Matrix<float, 3, 1> &c = C.col(i);
            glColor3f(color(0), color(1), color(2));
            // glColor3f(pos(0) / scale, pos(1) / scale, pos(2) / scale);

            glBegin(GL_POINTS);
            glVertex3f(pos(0) / scale, pos(1) / scale, pos(2) / scale);

            // if (draw_lines)
            //     glVertex3f(c(0) / scale, c(1) / scale, c(2) / scale);

            glEnd();

            if (draw_lines)
            {
                glBegin(GL_LINE_STRIP);
                glVertex3f(pos(0) / scale, pos(1) / scale, pos(2) / scale);
                glVertex3f(0, 0, 0);

                glVertex3f(0, 0, 0);
                glVertex3f(2 * c(0) / scale, 2 * c(1) / scale, 2 * c(2) / scale);

                glVertex3f(pos(0) / scale, pos(1) / scale, pos(2) / scale);
                glVertex3f(2 * c(0) / scale, 2 * c(1) / scale, 2 * c(2) / scale);

                glEnd();

                // std::cerr << "INFO: (r1+r2)/2  " << ((pos).norm() + (pos - 2 * c).norm()) / 2. << std::endl;
            }
        }
    }

    int numberOfOrbits = 100;
    Eigen::Matrix<float, DIM, -1> U, V, C, state, colors;
    Eigen::Matrix<float, 1, -1> rate_multiplier;
    Eigen::Matrix<float, 1, -1> t;

    const double MEU = 3.986004418e5; // km3s-2
    bool draw_lines = false;
};
