#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Calculator.h"
#include <cstdio>
#include <algorithm>

template<class T>
__global__ void addKernel(T *c, const T *a, const T *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void CUDA::Grid3::GPUalloc() {
    if(isGPUalloc) {
        throw CUDA::Grid3GPUReallocEx();
    }

    cudaMalloc((void**) &GPUdata, dataSize);

    isGPUalloc = true;
}

void CUDA::Grid3::GPUfree() {
    if(isGPUalloc) {
        throw CUDA::Grid3GPUFreeEx();
    }

    cudaFree(GPUdata);
}

void CUDA::Grid3::cpyDataFromGPU() {
    if(!isGPUalloc) {
        throw CUDA::Grid3WrongCallEx();
    }
    cudaMemcpy(GPUdata, data, dataSize, cudaMemcpyDeviceToHost);
}

void CUDA::Grid3::cpyDataToGPU() {
    if(!isGPUalloc) {
        throw CUDA::Grid3WrongCallEx();
    }
    cudaMemcpy(data, GPUdata, sizeof(double), cudaMemcpyHostToDevice);
}

void CUDA::addInt(int *c, const int *a, const int *b, unsigned int size) {
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

    cudaMalloc((void**) &dev_c, size * sizeof(int));
    cudaMalloc((void**) &dev_a, size * sizeof(int));
    cudaMalloc((void**) &dev_b, size * sizeof(int));

    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    addKernel<int><<<1, size>>>(dev_c, dev_a, dev_b);

    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}