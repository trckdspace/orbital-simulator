#pragma once

#define EIGEN_NO_CUDA

#include <Eigen/Eigen>
#include <GL/gl.h>

#include <chrono>
#include <thread>
#include <iostream>

#include <SimulatorBase.hpp>

#include <CollisionMap.hpp>

template <int DIM = 3>
struct Simulation : BaseSimulator
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    const float radius_of_earth_km = 6378.1;

    typedef Eigen::Array<float, -1, -1> MatrixType;

    Simulation(int orbit_count = 100) : BaseSimulator(orbit_count)
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

            float perigee = (float(rand() % 500) + (radius_of_earth_km + 160.f));
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

        state = C + Eigen::cos(t).replicate(DIM, 1) * U + Eigen::sin(t).replicate(DIM, 1) * V;
    }

    virtual void propagate(double dt)
    {
        static double clock_time = 0;
        clock_time += dt;

        t += speed * dt * rate_multiplier / (state).colwise().squaredNorm().array();
        state = C + Eigen::cos(t).replicate(DIM, 1) * U + Eigen::sin(t).replicate(DIM, 1) * V;

        CollisionDetector m;
        std::vector<std::pair<int, int>> collisions;
        if (m.run(state, collisions))
        {
            oh_no_these_collided.clear();
            for (auto c : collisions)
            {
                std::cerr << clock_time << " " << c.first << " " << c.second << std::endl;
                std::cerr << state.col(c.first).transpose() << std::endl;
                std::cerr << state.col(c.second).transpose() << std::endl;

                // std::cerr << " U " << U.col(c.first).transpose() << std::endl
                //           << " V " << V.col(c.first).transpose() << std::endl
                //           << " C " << C.col(c.first).transpose() << std::endl
                //           << " t " << t.col(c.first) << std::endl;

                // std::cerr << " U " << U.col(c.second).transpose() << std::endl
                //           << " V " << V.col(c.second).transpose() << std::endl
                //           << " C " << C.col(c.second).transpose() << std::endl
                //           << " t " << t.col(c.second) << std::endl;

                oh_no_these_collided.push_back(c.first);
                oh_no_these_collided.push_back(c.second);
            }
            // std::cerr << "//////////////////" << std::endl;
        }

        // std::cerr << state.rowwise().minCoeff() << std::endl;
        // std::cerr << state.rowwise().maxCoeff() << std::endl;
    }

    virtual void draw(int point_size, int count)
    {
        double scale = radius_of_earth_km;

        Eigen::ArrayXXf state_scaled = state / scale;

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        glVertexPointer(3, GL_FLOAT, sizeof(float) * 3, (float *)state_scaled.data());
        glColorPointer(3, GL_FLOAT, sizeof(float) * 3, colors.data());
        glPointSize(point_size);
        glDrawArrays(GL_POINTS, 0, std::min(count, (int)state.cols()));
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);

        // glFlush();
        //  glutSwapBuffers();

        if (draw_lines)
        {
            for (int i = 0; i < std::min(count, numberOfOrbits); i++)
            {
                const Eigen::Matrix<float, 3, 1> &pos = state.col(i);
                const Eigen::Matrix<float, 3, 1> &color = colors.col(i);
                const Eigen::Matrix<float, 3, 1> &c = C.col(i);

                // // glColor3f(pos(0) / scale, pos(1) / scale, pos(2) / scale);

                // glBegin(GL_POINTS);
                // glVertex3f(pos(0) / scale, pos(1) / scale, pos(2) / scale);

                // // if (draw_lines)
                // //     glVertex3f(c(0) / scale, c(1) / scale, c(2) / scale);

                // glEnd();
                glColor3f(color(0), color(1), color(2));

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

        for (auto i : oh_no_these_collided)
        {
            glPointSize(10);

            glColor3f(1, 0, 0);
            glBegin(GL_POINTS);

            for (float t = 0; t < 2 * M_PI; t += 0.1)
            {
                Eigen::ArrayXf _t = Eigen::ArrayXf::Ones(1, 1);
                Eigen::Vector3f v = C.col(i) + Eigen::cos(_t * t).replicate(DIM, 1) * U.col(i) + Eigen::sin(_t * t).replicate(DIM, 1) * V.col(i);
                glVertex3f(v(0) / scale, v(1) / scale, v(2) / scale);
            }
            const Eigen::Matrix<float, 3, 1> &pos = state.col(i);
            glColor3f(0, 0, 1);
            glVertex3f(pos(0) / scale, pos(1) / scale, pos(2) / scale);
            glEnd();
        }
    }

    MatrixType U, V, C, state, colors;
    Eigen::Array<float, 1, -1> rate_multiplier;
    Eigen::Array<float, 1, -1> t;

    const double MEU = 3.986004418e5; // km3s-2

    std::vector<int> oh_no_these_collided;
};
