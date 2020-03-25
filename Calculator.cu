#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Calculator.h"
#include <cstdio>
#include <algorithm>

void CUDA::Grid3::GPUalloc() {
    if(isGPUalloc) {
        throw CUDA::Grid3GPUReallocEx();
    }

    cudaMalloc((void**) &GPUdata, dataSize);

    isGPUalloc = true;
}

void CUDA::Grid3::GPUfree() {
    if(!isGPUalloc) {
        throw CUDA::Grid3GPUFreeEx();
    }

    cudaFree(GPUdata);
}

void CUDA::Grid3::cpyDataFromGPU() {
    if(!isGPUalloc) {
        throw CUDA::Grid3WrongCallEx();
    }
    cudaMemcpy(data, GPUdata, dataSize, cudaMemcpyDeviceToHost);
}

void CUDA::Grid3::cpyDataToGPU() {
    if(!isGPUalloc) {
        throw CUDA::Grid3WrongCallEx();
    }
    cudaMemcpy(GPUdata, data, dataSize, cudaMemcpyHostToDevice);
}

void CUDA::Calculator::initTask() {
    cudaMalloc((void**) &task, param.SIZE * param.SIZE * param.SIZE_Z * sizeof(bool));
    _task = new bool[param.SIZE * param.SIZE * param.SIZE_Z];
}

void CUDA::Calculator::freeTask() {
    cudaFree(task);
    delete[] _task;
}

#define N(x, y, z) (z) * size_x * size_y + (y) * size_x + (x)

__global__ void solve(double *data, double *goal, bool* task, int size_x, int size_y, int size_z) {
    //printf("(%d, %d, %d) in (%d %d %d)\n", threadIdx.x, threadIdx.y, threadIdx.y, blockIdx.x, blockIdx.y, blockIdx.z);
    int x = threadIdx.x + blockIdx.x * CUDA::BLOCK_SIZE;
    int y = threadIdx.y + blockIdx.y * CUDA::BLOCK_SIZE;
    int z = threadIdx.z + blockIdx.z * CUDA::BLOCK_SIZE;
    
    if(x >= size_x || y >= size_y || z >= size_z)
        return;
    
    if(!task[N(x, y, z, size_x, size_y)])
        return;

    goal[N(x, y, z)] = (data[N(x+1, y, z)] + data[N(x-1, y, z)] +
                        data[N(x, y+1, z)] + data[N(x, y-1, z)] +
                        data[N(x, y, z+1)] + data[N(x, y, z-1)]) / 6;                  
}


void CUDA::Calculator::calcU() {
    //std::cout << "u = " << u->at(param.SIZE / 2, param.SIZE - 2, param.SIZE_Z / 2) << std::endl;
	prev_u->cpyDataToGPU();

	int N_OPERATION = param.SIZE * param.SIZE * param.SIZE_Z * log(1 / param.EPS);
    int boost = 1000;

    int n_tasks = param.SIZE * param.SIZE * param.SIZE_Z;
    int counter = 0;

    for(int x = 0; x < param.SIZE; x ++) {
	for(int y = 0; y < param.SIZE; y ++) {
	for(int z = 0; z < param.SIZE_Z; z ++) {

		if(border->at(x, y, z) == 0.0f) {
            _task[x + y * param.SIZE + z * param.SIZE * param.SIZE] = true;
            counter ++;
        }
        else {
            _task[x + y * param.SIZE + z * param.SIZE * param.SIZE] = false;
        }
	}}}

    cudaMemcpy((void*) task, (void*)_task, n_tasks * sizeof(bool), cudaMemcpyHostToDevice);

    std::cout << "tasks: " << counter << "/" << n_tasks << " operations: " << N_OPERATION << std::endl;
   
    dim3 dimBlock(param.SIZE / CUDA::BLOCK_SIZE + 1,
                  param.SIZE / CUDA::BLOCK_SIZE + 1,
                  param.SIZE_Z / CUDA::BLOCK_SIZE + 1);

    dim3 dimThread(CUDA::BLOCK_SIZE,
                   CUDA::BLOCK_SIZE,
                   CUDA::BLOCK_SIZE);

    //std::cout << "(" << dimBlock.x << ", " << dimBlock.y << ", " << dimBlock.z << ")" << std::endl;
    //std::cout << "(" << dimThread.x << ", "<< dimThread.y << ", " << dimThread.z << ")" << std::endl;

    for(int op = 0; op < N_OPERATION / boost; op++) {
        
        solve<<< dimBlock, dimThread >>>(u->getGPUdata(), prev_u->getGPUdata(), task, param.SIZE, param.SIZE, param.SIZE_Z);

        //cudaDeviceSynchronize();
        std::swap(u, prev_u);
    }
    u->cpyDataFromGPU();

    //std::cout << "u = " << u->at(param.SIZE / 2, param.SIZE - 2, param.SIZE_Z / 2) << std::endl;
}

template<class T>
__global__ void addKernel(T *c, const T *a, const T *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
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

    addKernel<int><<<1 , size>>>(dev_c, dev_a, dev_b);

    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}