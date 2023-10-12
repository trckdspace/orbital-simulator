#pragma once

#include <thread>
#include <chrono>
#include <functional>
#include <string>

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

struct BaseSimulator
{
    BaseSimulator(int N) : numberOfOrbits(N) {}
    virtual void draw(int point_size, int num_satellites) = 0;
    virtual void propagate(double delta_t) = 0;
    virtual std::string getTime() = 0;

    bool draw_lines = false;
    float speed = 1;
    int numberOfOrbits;
    const float radius_of_earth_km = 6378.1;
};