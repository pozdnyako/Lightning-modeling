#include "Matrix.h"
#include <cstdio>

Matrix::Matrix() {
    is_mem_allocated = false;
}

Matrix::Matrix(int y, int x) {
    sizeX = x;
    sizeY = y;

    allocate_mem();
}

Matrix::Matrix(const Matrix& matrix) {
    sizeX = matrix.sizeX;
    sizeY = matrix.sizeY;

    allocate_mem();

    for(int i = 0; i < sizeX * sizeY; i ++) {
        data[i] = matrix.data[i];
    }
}

void Matrix::allocate_mem() {
    if(is_mem_allocated) {
        delete[] data;
    }

    data = new double[sizeX * sizeY];
    is_mem_allocated = true;
}

int Matrix::x_p(int pos) {
    return pos % sizeX;
}

int Matrix::y_p(int pos) {
    return pos / sizeX;
}

int Matrix::p_xy(int x, int y) {

    //printf("\n%d %d -> %d\n", x, y, y * sizeX + x);
    return y * sizeX + x;
}

void Matrix::set_size(int y, int x) {
    if(is_mem_allocated) {
        delete[] data;
    }

    sizeX = x;
    sizeY = y;

    allocate_mem();
}

void Matrix::set_data(double* _data) {
    for(int i = 0; i < sizeX * sizeY; i ++) {
        data[i] = _data[i];
    }
}

void Matrix::set_num(int y, int x, double num) {
    data[p_xy(x,y)] = num;
}

double Matrix::get_num(int y, int x) {
    return data[p_xy(x,y)];
}

void Matrix::print() {
    printf("%dx%d \n", sizeY, sizeX);

    for(int y = 0; y < sizeY; y++) {
    for(int x = 0; x < sizeX; x ++) {

        printf("%5.2f", get_num(y, x));
    } printf("\n");}

    printf("\n\n");
}

Matrix::~Matrix() {
    if(is_mem_allocated) {
        delete[] data;
    }
}
