#include "Calculator.h"

CUDA::Calculator::Calculator(const Parameters &_param) :
	lst_chargeCount(0),
	borderSize(0) {
	param = _param;

	u1 = new CUDA::Grid3(param.SIZE, param.SIZE, param.SIZE_Z);
	u2 = new CUDA::Grid3(param.SIZE, param.SIZE, param.SIZE_Z);
	u = u1;
	prev_u = u2;

	border = new CUDA::Grid3(param.SIZE, param.SIZE, param.SIZE_Z, false);

	initBorder();
	initTask();
}

CUDA::Calculator::~Calculator() {
	delete u1;
	delete u2;
	delete border;

	freeTask();
}

void CUDA::Calculator::initBorder() {
	for(int x = 0; x < param.SIZE; x++) {
	for(int z = 0; z < param.SIZE_Z; z++) {
		
		border->set(1.0f, x, param.SIZE-1, z);
		border->set(1.0f, x, 0, z);
		
		u->set(param.U_0, x, param.SIZE-1, z);
	}}

	for(int y = 0; y < param.SIZE; y++) {
	for(int z = 0; z < param.SIZE_Z; z++) {
		border->set(1.0f, 0, y, z);
		border->set(1.0f, param.SIZE-1, y, z);
	}}

	for(int x = 0; x < param.SIZE; x++) {
	for(int y = 0; y < param.SIZE; y++) {
		border->set(1.0f, x, y, 0);
		border->set(1.0f, x, y, param.SIZE_Z-1);		
	}}

	borderSize = param.SIZE * param.SIZE * param.SIZE_Z -
				 (param.SIZE - 2) * (param.SIZE - 2) * (param.SIZE_Z - 2);
}

void CUDA::Calculator::updateBorder() {
	if(lst_chargeCount == charge.size())
		return;

	for(int i = lst_chargeCount - 1; i < charge.size(); i++) {
		border->set(1.0f, charge[i]);

		u->set(param.U_0, charge[i]);

		borderSize++;
	}
}











CUDA::Grid3::Grid3() :
	size(0, 0, 0),
	isGPUalloc(false),
	dataSize(size.x * size.y * size.z * sizeof(double)),
	data(NULL),
	GPUdata(NULL) {}

CUDA::Grid3::Grid3(int x, int y, int z, bool GPU) :
	size(x, y, z),
	isGPUalloc(false),
	dataSize(size.x * size.y * size.z * sizeof(double)),
	data(NULL),
	GPUdata(NULL),
	hasGPU(GPU){
	
	data = new double[x * y * z];

	for(int i = 0; i < x*y*z; i++) {
		data[i] = 0;
	}

	if(hasGPU)
		GPUalloc();
}

CUDA::Grid3::~Grid3() {
	if(size*size > 0)
		delete[] data;

	if(hasGPU)
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

void CUDA::Grid3::set(double num, const Vector3i &p) {
	if(is_in(p.x, 0, size.x - 1) &&
	   is_in(p.y, 0, size.y - 1) &&
	   is_in(p.z, 0, size.z - 1)) {

		data[p2n(p)] = num;
	}
	else
		throw CUDA::Grid3WrongCallEx();
}
