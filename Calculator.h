#pragma once

#include "cuda_runtime.h"
#include "Math.h"
#include "Parameters.h"
#include <vector>

namespace CUDA {
    class Grid3;

    void printDevicesProperties();
    const int BLOCK_SIZE = 8;

    const int N_DIR = 8;
    const std::vector<Vector3i> DIR{
        Vector3i( 0, 1, 0),
        Vector3i( 0,-1, 0),
        Vector3i( 1, 0, 0),
        Vector3i(-1, 0, 0),

        Vector3i( 1, 1, 0),
        Vector3i( 1,-1, 0),
        Vector3i(-1, 1, 0),
        Vector3i(-1,-1, 0),
    };


    class Calculator {
    public:
        Calculator():u1(NULL),u2(NULL),u(NULL),prev_u(NULL) {};
        Calculator(const Parameters &);
        ~Calculator();

        void calcU();
        void grow();

        const std::pair<int, int> newDir();
        const Vector3 calcE(const Vector3i&);

        double getU(int, int, int);
        double getU(const Vector3i& v) { return getU(v.x, v.y, v.z); };

        const std::vector<Vector3i>& getCharge() { return charge; }
    private:
        Parameters param;

        Grid3 *u1, *u2;
        Grid3 *u, *prev_u;

        Grid3 *border;
        std::vector<Vector3i> charge;

        bool * task;
        bool *_task;

        int borderSize;
        int lst_chargeCount;
        void initBorder();
        void updateBorder();

        void initTask();
        void freeTask();
    };

    class Grid3 {
    public:
        Grid3();
        Grid3(int, int, int, bool GPU=true);

        ~Grid3();

        double at(int, int, int);
        double at(const Vector3i& v) { return at(v.x, v.y, v.z); }
        void set(double, int, int, int);
        void set(double, const Vector3i&);

        void cpyDataToGPU();
        void cpyDataFromGPU();

        int p2n(const Vector3i&);
        int p2n(const int&, const int&, const int&);
        Vector3i n2p(const int&);

        double* getGPUdata() { return GPUdata; }
    private:
        double *data;
        double *GPUdata;

        int dataSize;

        Vector3i size;

        bool isGPUalloc;
        void GPUalloc();
        void GPUfree();

        bool hasGPU;
    };


    void addInt(int*, const int*, const int*, unsigned int);




    class Grid3WrongCallEx : public WrongCallEx {
    public:
        Grid3WrongCallEx(int _x, int _y, int _z, int __x, int __y, int __z):
            x(_x), y(_y), z(_z),
            x_max(__x), y_max(__y), z_max(__z) {}
        Grid3WrongCallEx() {}

        std::string what() {
            return WrongCallEx::what() + ", Grid3 at ("
                 + std::to_string(x) + ", " 
                 + std::to_string(y) + ", "
                 + std::to_string(z) + ") in ("
                 + std::to_string(x_max) + ", "
                 + std::to_string(y_max) + ", "
                 + std::to_string(z_max) + ") ";
        }
    private:
        int x, y, z;
        int x_max, y_max, z_max;
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