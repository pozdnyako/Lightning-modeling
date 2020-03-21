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
    cudaMemcpy(GPUdata, data, dataSize, cudaMemcpyDeviceToHost);
}

void CUDA::Grid3::cpyDataToGPU() {
    if(!isGPUalloc) {
        throw CUDA::Grid3WrongCallEx();
    }
    cudaMemcpy(data, GPUdata, sizeof(double), cudaMemcpyHostToDevice);
}

void CUDA::Calculator::initTask() {
    cudaMalloc((void**) &task, param.SIZE * param.SIZE * param.SIZE_Z * sizeof(CUDA::Task));
}

void CUDA::Calculator::freeTask() {
    cudaFree(task);
}

__global__ void solve(double *data, double *goal, CUDA::Task* task) {
    int i = threadIdx.x;

    CUDA::Task* cur = task + i;

    goal[cur->a] = (data[cur->a1] + data[cur->a2] + data[cur->a3] +
                    data[cur->a4] + data[cur->a5] + data[cur->a6]) / 6;
}


void CUDA::Calculator::calcU() {
	u->cpyDataToGPU();

	int N_OPERATION = param.SIZE * param.SIZE * param.SIZE_Z / 10 * log(1 / param.EPS);
	int boost = 1000;

    int n_tasks = param.SIZE * param.SIZE * param.SIZE_Z - borderSize;

    int counter = 0;

    Task cur_task;
	for(int x = 1; x < param.SIZE - 1; x ++) {
	for(int y = 1; y < param.SIZE - 1; y ++) {
	for(int z = 1; z < param.SIZE_Z - 1; z ++) {

		if(border->at(x, y, z) == 0.0f) {
            cur_task.a  = border->p2n(x, y, z);

            cudaMemcpy(&cur_task, (void*) (task+counter), sizeof(Task), cudaMemcpyHostToDevice);

            counter++;
        }
	}}}

    std::cout << "tasks: " << n_tasks << std::endl;

    for(int op = 0; op < N_OPERATION; op++) {
        solve<<<1, n_tasks>>>(u->getGPUdata(), prev_u->getGPUdata(), task);

        std::swap(u, prev_u);
    }
    cudaDeviceSynchronize();
    std::cout << "finish" << std::endl;
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

    addKernel<int><<<1, size>>>(dev_c, dev_a, dev_b);

    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
}