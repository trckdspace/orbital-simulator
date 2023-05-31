#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char *const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

struct Position
{
    float x, y, z;
};

struct Orbit
{
    typedef double Scalar;

    Scalar u0, u1, u2;
    Scalar v0, v1, v2;
    Scalar c0, c1, c2;
    double rate_multiplier;
    Scalar t;

__host__ __device__ 
    void propagate(Position& p, Orbit::Scalar delta_t, bool is_init = false)
    {
        if (!is_init)
        {
            Scalar r_2 = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
            t = t + 100* delta_t * (rate_multiplier / (r_2 * r_2));
        }

        Scalar st = std::sin(t);
        Scalar ct = std::cos(t);

        p.x = ct * u0 + st * v0 + c0;
        p.y = ct * u1 + st * v1 + c1;
        p.z = ct * u2 + st * v2 + c2;
    }
};

typedef Orbit Orbit;



__global__ void step(
    Orbit *satellites,
    Position *positions,
    int width,
    int height,
    double delta_t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    auto at = [width](int x, int y)
    { return y * width + x; };

    if (x < 0 or x > width)
        return;
    if (y < 0 or y > height)
        return;

    int idx = at(x, y);
    Orbit &sat = satellites[idx];
    Position &pos = positions[idx];
    sat.propagate(pos, delta_t, false);
}
