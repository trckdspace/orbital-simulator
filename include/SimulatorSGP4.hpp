#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

#include <date.h>

#include <perturb/perturb.hpp>
#include "SimulatorBase.hpp"

#include <GL/gl.h>

std::ostream &operator<<(std::ostream &os, const std::array<double, 3> &a)
{
    os << a[0] << " " << a[1] << " " << a[2];
    return os;
}

std::array<double, 3> operator-(const std::array<double, 3> a, const std::array<double, 3> b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

double dot(const std::array<double, 3> &a, const std::array<double, 3> &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<double, 3> normalize(const std::array<double, 3> &v)
{
    double divisor = dot(v, v);
    return {v[0] / divisor, v[1] / divisor, v[2] / divisor};
}

struct CollisionDetector
{
    // Given a current position vector, quantize and fill bins to detect collisions.

    double distance(const perturb::StateVector &a, const perturb::StateVector &b)
    {
        return std::sqrt(
            (a.position[0] - b.position[0]) * (a.position[0] - b.position[0]) +
            (a.position[1] - b.position[1]) * (a.position[1] - b.position[1]) +
            (a.position[2] - b.position[2]) * (a.position[2] - b.position[2]));
    }

    double angle(const perturb::StateVector &a, const perturb::StateVector &b)
    {
        auto a_n = normalize(a.velocity);
        auto b_n = normalize(b.velocity);

        return std::acos(dot(a_n, b_n));
    }

    double rel_vel(const perturb::StateVector &a, const perturb::StateVector &b)
    {
        auto diff = a.velocity - b.velocity;
        return std::sqrt(dot(diff, diff));
    }

    bool run(const std::vector<perturb::StateVector> &positions, std::vector<std::pair<int, int>> &collisions)
    {
        for (int i = 0; i < positions.size(); i++)
        {
            auto s = positions[i];
            auto d = std::sqrt(s.position[0] * s.position[0] + s.position[0] * s.position[0] + s.position[0] * s.position[0]);
            uint64_t x = int(std::abs(s.position[0])) / 10;
            uint64_t y = int(std::abs(s.position[1])) / 10;
            uint64_t z = int(std::abs(s.position[2])) / 10;

            uint64_t idx = (x << 32) + (y << 16) + z;

            position_bins[idx].push_back(i);
        }

        for (const auto &bin : position_bins)
        {

            if (bin.second.size() < 2)
                continue;
            // std::cerr << bin.first << " : ";

            for (size_t i = 0; i < bin.second.size(); i++)
            {
                auto p = bin.second[i];
                for (size_t j = i + 1; j < bin.second.size(); j++)
                {
                    auto q = bin.second[j];
                    double dist = distance(positions[p], positions[q]);
                    double angle_rads = M_PI / 180 * 15;
                    if (dist > 5 or rel_vel(positions[p], positions[q]) < 10)
                        continue;
                    // std::cerr << "(" << p << "," << q << "," << dist << ", " << angle(positions[p], positions[q]) << ")" << std::endl;

                    collisions.push_back({p, q});
                }

                // std::cerr << std::endl;
            }
        }

        return !collisions.empty();
    }

    std::unordered_map<uint64_t, std::vector<uint32_t>> position_bins;
};

struct SimulatorSGP4 : BaseSimulator
{
    SimulatorSGP4(const char *filename) : BaseSimulator(-1)
    {
        std::ifstream in(filename);

        char name[80];
        char line1[80];
        char line2[80];

        std::vector<int> names;

        while (in.is_open() and in.good())
        {
            in.getline(name, sizeof(name));
            in.getline(line1, sizeof(line1));
            in.getline(line2, sizeof(line2));

            perturb::StateVector v;

            if (in.good())
            {
                auto s = perturb::Satellite::from_tle(line1, line2);

                for (auto n : names)
                    if (n == atoi(s.sat_rec.satnum))
                    {
                        continue;
                    }

                names.push_back(atoi(s.sat_rec.satnum));
                // std::cerr << s.sat_rec.classification << std::endl;
                if (std::string(name).find("DEB") == std::string::npos)
                    colors.push_back(COLOR{0, 0, 1});
                else
                    colors.push_back(COLOR{0.5, 0.5, 0.5});

                // s.propagate_from_epoch(0, v);
                // const auto &pos = v.position;

                // auto d = std::sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);

                // if (d > radius_of_earth_km + 500)
                //     continue;

                if (s.last_error() == perturb::Sgp4Error::NONE)
                {
                    satellites.push_back(s);
                }
            }
        }

        in.close();

        states.resize(satellites.size());

        numberOfOrbits = satellites.size();
        start_time = std::chrono::system_clock::now();
    }

    virtual std::string getTime()
    {
        char buffer[64];
        sprintf(buffer, "%04d:%02d:%02dT%02d:%02d:%02d.%02d",
                dt.year,
                dt.month,
                dt.day,
                dt.hour,
                dt.min, int(dt.sec), int(100 * (dt.sec - int(dt.sec))));
        return std::string(buffer);
    }

    virtual void draw(int point_size, int num_satellites)
    {
        double scale = radius_of_earth_km;
        glPointSize(point_size);
        glBegin(GL_POINTS);
        //  for (auto s : states)
        for (int i = 0; i < states.size(); i++)
        {
            auto &s = states[i];
            glColor3f(colors[i].r, colors[i].g, colors[i].b);
            glVertex3f(s.position[0] / scale, s.position[1] / scale, s.position[2] / scale);
        }
        glEnd();

        glPointSize(10);
        glColor3f(1, 0, 0);
        glBegin(GL_POINTS);

        for (auto i : oh_no_these_collided)
        {
            auto &p = states[i].position;
            glVertex3f(p[0] / scale, p[1] / scale, p[2] / scale);
        }
        glEnd();
    }

    virtual void
    propagate(double delta_t)
    {
        start_time += std::chrono::milliseconds(int(1000 * delta_t * speed));
        auto dp = date::floor<date::days>(start_time);
        auto ymd = date::year_month_day{dp};
        auto time = date::make_time(std::chrono::duration_cast<std::chrono::milliseconds>(start_time - dp));

        // std::cout << "year        = " << ymd.year() << '\n';
        // std::cout << "month       = " << ymd.month() << '\n';
        // std::cout << "day         = " << ymd.day() << '\n';
        // std::cout << "hour        = " << time.hours().count() << "h\n";
        // std::cout << "minute      = " << time.minutes().count() << "min\n";
        // std::cout << "second      = " << time.seconds().count() << "s\n";
        // std::cout << "millisecond = " << time.subseconds().count() << "ms\n";
        const auto t = perturb::JulianDate(perturb::DateTime{(int)ymd.year(),
                                                             (unsigned int)ymd.month(),
                                                             (unsigned int)ymd.day(),
                                                             (int)time.hours().count(),
                                                             (int)time.minutes().count(),
                                                             (float)time.seconds().count() + time.subseconds().count() / 1000.});

        dt = t.to_datetime();
        std::cerr << dt.year << ":" << dt.month << ":" << dt.day << "T" << dt.hour << ":" << dt.min << ":" << dt.sec << std::endl;
        for (size_t i = 0; i < satellites.size(); i++)
        {
            if (satellites[i].last_error() == perturb::Sgp4Error::NONE)
                satellites[i].propagate(t, states[i]);
        }

        CollisionDetector detector;
        std::vector<std::pair<int, int>> collisions;

        if (detector.run(states, collisions))
        {
            oh_no_these_collided.clear();
            for (auto c : collisions)
            {
                oh_no_these_collided.push_back(c.first);
                oh_no_these_collided.push_back(c.second);

                std::cerr << "CDM EVENT: " << satellites[c.first].sat_rec.satnum << " " << satellites[c.second].sat_rec.satnum << std::endl;
                std::cerr << "Sat 1" << std::endl;
                std::cerr << "\t" << states[c.first].position << " " << states[c.first].velocity << std::endl;
                std::cerr << "Sat 2" << std::endl;
                std::cerr << "\t" << states[c.second].position << " " << states[c.second].velocity << std::endl;
            }
        }
    }

    struct COLOR
    {
        GLfloat r, g, b;
    };

    std::vector<perturb::Satellite> satellites;
    std::vector<perturb::StateVector> states;
    std::vector<COLOR> colors;

    float seconds = 0;
    std::chrono::time_point<std::chrono::system_clock> start_time;
    // std::vector<std::pair<int, int>> collisions;
    std::vector<int> oh_no_these_collided;

    perturb::DateTime dt;
};