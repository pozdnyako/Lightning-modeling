#include "Matrix3.h"
#include <cstdio>

Matrix3::Matrix3() {
    is_mem_allocated = false;
}

Matrix3::Matrix3(int y, int x, int z) {
    is_mem_allocated = false;

    sizeX = x;
    sizeY = y;
    sizeZ = z;

    allocate_mem();
}

Matrix3::Matrix3(const Matrix3& matrix) {
    is_mem_allocated = false;

    sizeX = matrix.sizeX;
    sizeY = matrix.sizeY;
    sizeZ = matrix.sizeZ;

    allocate_mem();

    for(int i = 0; i < sizeX * sizeY * sizeZ; i ++) {
        data[i] = matrix.data[i];
    }
}

void Matrix3::allocate_mem() {
    if(is_mem_allocated) {
        delete[] data;
    }

    data = new double[sizeX * sizeY * sizeZ];
    is_mem_allocated = true;
}

int Matrix3::x_p(int pos) {
    return pos % sizeZ % sizeX;
}

int Matrix3::y_p(int pos) {
    return pos % sizeZ / sizeX;
}

int Matrix3::z_p(int pos) {
    return pos / sizeZ;
}

int Matrix3::p_xyz(int x, int y, int z) {

    //printf("\n%d %d -> %d\n", x, y, y * sizeX + x);
    return (y * sizeX + x) * sizeZ + z;
}

void Matrix3::set_size(int y, int x, int z) {
    if(is_mem_allocated) {
        delete[] data;
    }

    sizeX = x;
    sizeY = y;
    sizeZ = z;

    allocate_mem();
}

void Matrix3::set_data(double* _data) {
    for(int i = 0; i < sizeX * sizeY * sizeZ; i ++) {
        data[i] = _data[i];
    }
}

void Matrix3::set_num(int y, int x, int z, double num) {
    data[p_xyz(x,y,z)] = num;
}

double Matrix3::get_num(int y, int x, int z) {
    return data[p_xyz(x,y,z)];
}

void Matrix3::print(int z) {
    printf("%dx%dx%d \n", sizeY, sizeX, sizeZ);

    for(int y = 0; y < sizeY; y++) {
    for(int x = 0; x < sizeX; x ++) {

        printf("%5.2f", get_num(y, x, z));
    } printf("\n");}

    printf("\n\n");
}

Matrix3::~Matrix3() {
    if(is_mem_allocated) {
        delete[] data;
    }
}
