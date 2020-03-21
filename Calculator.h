#pragma once

#include "cuda_runtime.h"
#include "Math.h"
#include "Parameters.h"
#include <vector>

namespace CUDA {
    class Grid3;

    class Calculator {
    public:
        Calculator():u1(NULL),u2(NULL),u(NULL) {};
        Calculator(const Parameters &);
        ~Calculator();
    private:

        Grid3 *u1, *u2;
        Grid3 *u;
    };

    class Grid3 {
    public:
        Grid3();
        Grid3(int, int, int);

        ~Grid3();

        double at(int, int, int);
        void set(double, int, int, int);

        void cpyDataToGPU();
        void cpyDataFromGPU();
    private:
        double *data;
        double *GPUdata;

        int dataSize;

        Vector3i size;

        int p2n(const Vector3i&);
        int p2n(const int&, const int&, const int&);
        Vector3i n2p(const int&);

        bool isGPUalloc;
        void GPUalloc();
        void GPUfree();
    };


    void addInt(int*, const int*, const int*, unsigned int);


    class Grid3WrongCallEx : public WrongCallEx {
    public:
        std::string what() {
            return WrongCallEx::what() + ", Grid3";
        }
    };

    class Grid3GPUReallocEx : public AllocEx {
    public:
        std::string what() {
            return AllocEx::what() + ", gpu realloc";
        }
    };

    class Grid3GPUFreeEx : public FreeEx {
    public:
        std::string what() {
            return FreeEx::what() + ", gpu refree";
        }
    };
}