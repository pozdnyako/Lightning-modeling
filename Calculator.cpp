#include "Calculator.h"

CUDA::Calculator::Calculator(const Parameters &param) {
	u1 = new CUDA::Grid3(param.SIZE, param.SIZE, param.SIZE_Z);
	u2 = new CUDA::Grid3(param.SIZE, param.SIZE, param.SIZE_Z);
	
	u = u1;
}

CUDA::Calculator::~Calculator() {
	delete u1;
	delete u2;
}

CUDA::Grid3::Grid3() :
	size(0, 0, 0), isGPUalloc(false), dataSize(size.x * size.y * size.z * sizeof(double)){}

CUDA::Grid3::Grid3(int x, int y, int z) :
	size(x, y, z), isGPUalloc(false), dataSize(size.x * size.y * size.z * sizeof(double)) {
	
	data = new double[x * y * z];

	GPUalloc();
}

CUDA::Grid3::~Grid3() {
	if(size*size > 0)
		delete[] data;

	GPUfree();
}

int CUDA::Grid3::p2n(const Vector3i& p) {
	return p.x + p.y * size.x + p.z * size.x * size.y;
}

int CUDA::Grid3::p2n(const int& x, const int& y, const int& z) {
	return x + y * size.x + z * size.x * size.y;
}
Vector3i CUDA::Grid3::n2p(const int& n) {
	return Vector3i(n % size.x, n / size.x % size.y, n / size.x / size.y);
}


double CUDA::Grid3::at(int x, int y, int z) {
	if(is_in(x, 0, size.x - 1) &&
	   is_in(y, 0, size.y - 1) &&
	   is_in(z, 0, size.z - 1)) {

		return data[p2n(x, y, z)];
	}
	else
		throw CUDA::Grid3WrongCallEx();
}

void CUDA::Grid3::set(double num, int x, int y, int z) {
	if(is_in(x, 0, size.x - 1) &&
	   is_in(y, 0, size.y - 1) &&
	   is_in(z, 0, size.z - 1)) {

		data[p2n(x,y,z)] = num;
	}
	else
		throw CUDA::Grid3WrongCallEx();
}