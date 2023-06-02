#pragma once

/*
Structure to abstract away dealing with aligning memory each time we need to use something on the GPU

1. Aligns both CPU and GPU memory on init (this may not always be needed)
2. Can accept external data for host side of thing and copies that to GPU in constructor ONLY.
3. Cleans up after itself in the destructor.

*/

template <typename Scalar>
struct MemoryBlock
{
    size_t width;
    size_t height;
    size_t size;

    Scalar *device_ptr = nullptr;
    Scalar *host_ptr = nullptr;

    bool ownsHostData = false;

    MemoryBlock(size_t w = 1, size_t h = 1) : width(w), height(h), size(w * h), ownsHostData(true)
    {
        this->allocate_on_host();
        this->allocate_on_device();
    }

    MemoryBlock(size_t w, size_t h, Scalar *data) : width(w), height(h), size(w * h), host_ptr(data), ownsHostData(false)
    {
        this->allocate_on_device();
        this->copy_to_device();
    }

    MemoryBlock(Scalar *data) : MemoryBlock(1, 1, data) {}

    bool allocate_on_device()
    {
        cudaError_t err = cudaMalloc((void **)&device_ptr, width * height * sizeof(Scalar));
        // cudaError_t err = cudaMallocManaged((void **)&device_ptr, width * height * sizeof(Scalar));
        cudaMemset(device_ptr, 0x00, width * height * sizeof(Scalar));
        return err == cudaError_t::cudaSuccess;
    }

    void setZero_device()
    {
        if (this->isAllocatedGPU())
            cudaMemset(device_ptr, 0x00, width * height * sizeof(Scalar));
    }

    bool allocate_on_host()
    {
        if (!host_ptr)
            host_ptr = new Scalar[width * height];
        return true;
    }

    bool copy_to_host()
    {
        if (host_ptr)
        {
            auto err = cudaMemcpy((void *)host_ptr, (void *)device_ptr, width * height * sizeof(Scalar), cudaMemcpyDeviceToHost);
            return err == cudaError_t::cudaSuccess;
        }
        return cudaError_t::cudaErrorCapturedEvent;
    }

    bool copy_to_device()
    {
        auto err = cudaMemcpy((void *)device_ptr, (void *)host_ptr, width * height * sizeof(Scalar), cudaMemcpyHostToDevice);
        return err == cudaError_t::cudaSuccess;
    }

    bool isAllocatedGPU() { return device_ptr != nullptr; }

    bool isAllocatedCPU() { return host_ptr != nullptr; }

    ~MemoryBlock()
    {
        if (this->isAllocatedGPU())
            cudaFree(device_ptr);

        if (this->isAllocatedCPU() and ownsHostData)
        {
            delete[] host_ptr;
            host_ptr = nullptr;
        }
    }
};

template <typename Scalar>

struct MemoryBlockManaged
{

    size_t width;
    size_t height;
    size_t size;
    Scalar *device_ptr = nullptr;
    Scalar *host_ptr = nullptr;

    bool ownsHostData = false;

    MemoryBlockManaged(size_t w = 1, size_t h = 1) : width(w), height(h), size(w * h), ownsHostData(true)
    {
        this->allocate(width, height);
        this->host_ptr = this->device_ptr;
    }

    bool allocate(int width, int height)
    {
        // cudaError_t err = cudaMalloc((void **)&device_ptr, width * height * sizeof(Scalar));
        cudaError_t err = cudaMallocManaged((void **)&device_ptr, width * height * sizeof(Scalar));
        cudaMemset(device_ptr, 0x00, width * height * sizeof(Scalar));
        return err == cudaError_t::cudaSuccess;
    }

    void setZero_device()
    {
        cudaMemset(device_ptr, 0x00, width * height * sizeof(Scalar));
    }

    ~MemoryBlockManaged()
    {
        cudaFree(device_ptr);
    }
};
