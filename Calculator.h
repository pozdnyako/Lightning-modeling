#pragma once

#include "cuda_runtime.h"

class Calculator {
public:

    static cudaError_t addWithCuda(int *, const int *, const int *, unsigned int);
private:
    cudaError_t staus;
};