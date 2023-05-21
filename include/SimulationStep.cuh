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

template <typename Scalar>
struct Orbit
{
    Scalar u0, u1, u2;
    Scalar v0, v1, v2;
    Scalar t;
    Scalar v;
};

typedef Orbit<float> Orbit_f;

struct Result
{
    float x, y, z;
};

__global__ void step(
    Orbit_f *satellites,
    Result *positions,
    int width,
    int height,
    double delta_t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    auto at = [width](int x, int y)
    { return y * width + x; };

    if (x < 0 or x >= width)
        return;
    if (y < 0 or y >= height)
        return;

    int idx = at(x, y);
    Orbit_f &sat = satellites[idx];
    Result &res = positions[idx];

    // https: // www.physicsclassroom.com/class/circles/Lesson-4/Mathematics-of-Satellite-Motion
    // http: // www.physicsclassroom.com/Class/circles/u6l4b5.gif v = sqrt(G*M/R)
    // gravitational constant *mass of Earth = 3.98601877 × 10^14 m^3 / s^2

    sat.t += delta_t;

    float st = std::sin(sat.t * sat.v);
    float ct = std::cos(sat.t * sat.v);

    float px = st * sat.u0 + ct * sat.v0;
    float py = st * sat.u1 + ct * sat.v1;
    float pz = st * sat.u2 + ct * sat.v2;

    float R_m = std::sqrt(px * px + py * py + pz * pz);
    sat.v = (std::sqrt(3.98601877e14 / (R_m))) / R_m;

    res.x = px;
    res.y = py;
    res.z = pz;
}
