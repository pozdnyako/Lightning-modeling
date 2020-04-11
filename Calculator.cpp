#include "Calculator.h"

std::ostream& operator<<(std::ostream &os, const cudaDeviceProp &prop) {
    os << prop.name << ":\n" << std::endl;
    os << "\t" << prop.totalGlobalMem << " - total amount of memory on the device in bytes" << std::endl;
	os << "\t" << prop.sharedMemPerBlock<< " - the maximum amount of shared memory available to a thread block in bytes" << std::endl;
	os << "\t" << prop.warpSize << " -  the warp size in threads\n";
	os << "\t" << prop.maxThreadsPerBlock << " - maximum numbers of threads in blocks\n";
	os << "\t" << prop.maxThreadsDim[0] << "x"
		<< prop.maxThreadsDim[1] << "x"
		<< prop.maxThreadsDim[2] << " - maximum size of each dimension in blocks\n";

	os << "\t" << prop.clockRate/1000 << " MHz - clock freq\n";

	return os;
}

void CUDA::printDevicesProperties() {
    int n_dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceCount(&n_dev);
    
    for(int i = 0; i < n_dev; i++) {
        cudaGetDeviceProperties(&prop, i);
        std::cout << prop;
    }
}

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

	charge.push_back(Vector3i(param.SIZE / 2, param.SIZE - 2, param.SIZE_Z / 2));
	charge.push_back(Vector3i(param.SIZE / 2, param.SIZE - 3, param.SIZE_Z / 2));
	charge.push_back(Vector3i(param.SIZE / 2, param.SIZE - 4, param.SIZE_Z / 2));
	updateBorder();
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
		prev_u->set(param.U_0, x, param.SIZE-1, z);
	}}

	for(int y = 0; y < param.SIZE; y++) {
	for(int z = 0; z < param.SIZE_Z; z++) {
		border->set(1.0f, 0, y, z);
		border->set(1.0f, param.SIZE-1, y, z);

		u->set(param.U_0 * y / param.SIZE, 0, y, z);
		u->set(param.U_0 * y / param.SIZE, param.SIZE-1, y, z);

		prev_u->set(param.U_0 * y / param.SIZE, 0, y, z);		
		prev_u->set(param.U_0 * y / param.SIZE, param.SIZE-1, y, z);
	}}

	for(int x = 0; x < param.SIZE; x++) {
	for(int y = 0; y < param.SIZE; y++) {
		border->set(1.0f, x, y, 0);
		border->set(1.0f, x, y, param.SIZE_Z-1);	

		u->set(param.U_0 * y / param.SIZE, x, y, 0);
		u->set(param.U_0 * y / param.SIZE, x, y, param.SIZE_Z-1);
	
		prev_u->set(param.U_0 * y / param.SIZE, x, y, 0);
		prev_u->set(param.U_0 * y / param.SIZE, x, y, param.SIZE_Z-1);
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
		prev_u->set(param.U_0, charge[i]);

		borderSize++;
	}

	lst_chargeCount = charge.size();
}

double CUDA::Calculator::getU(int x, int y, int z) {
	return u->at(x, y, z);
}


const Vector3 CUDA::Calculator::calcE(const Vector3i& v) {
    return Vector3(getU(v + Vector3i(-1, 0, 0)) - getU(v + Vector3i(1, 0, 0)),
                   getU(v + Vector3i( 0,-1, 0)) - getU(v + Vector3i(0, 1, 0)),
                   getU(v + Vector3i( 0, 0,-1)) - getU(v + Vector3i(0, 0, 1))) * 0.5f;
}

const std::pair<int, int> CUDA::Calculator::newDir() {
    std::vector<std::pair<int, int> > goals;

    for(int i = 0; i < charge.size(); i++) {
        for(int j = 0; j < CUDA::N_DIR; j++) {
            Vector3i goal = charge[i] + DIR[j];

            if(border->at(goal) == 1.0f)
                continue;

            goals.push_back(std::make_pair(i, j));
        }
    }

	//std::cout << "grow directions: " << goals.size() << std::endl;

    std::vector<double> prob;
    double prob_sum = 0.0f;
	double max_prob = 0.0f;

    for(int i = 0; i < goals.size(); i++) {
        double E = calcE(charge[goals[i].first] + Vector3(DIR[goals[i].second])) * Vector3(DIR[goals[i].second]) / DIR[goals[i].second].getNorm() ;
		double cur_prob = 0.0f;


		if(E > 2) {
			cur_prob = pow(E-2, 2);
			//std::cout << calcE(charge[goals[i].first] + Vector3(DIR[goals[i].second])) << " * " << Vector3(DIR[goals[i].second]) << "=" << E << std::endl;
		}

		if(max_prob < cur_prob)
			max_prob = cur_prob;

		prob_sum += cur_prob;
        prob.push_back(cur_prob);
    }

	if(prob_sum == 0.0f)
		std::cout << "WRONG prob" << std::endl;


    double res = ((double)rand() / RAND_MAX) * prob_sum;

    int res_n = 0;

    do {
        res -= prob[res_n];
        res_n ++;
	} while( (res > 0.0f || prob[res_n] == 0.0f) && res_n < goals.size() - 1);

	std::cout << "prob: " << prob[res_n] / prob_sum << " (max = "<< max_prob / prob_sum << ")" << std::endl;

    return goals[res_n];
}

void CUDA::Calculator::grow() {
    std::pair<int, int> dir_pair = newDir();

    charge.push_back(charge[dir_pair.first] + DIR[dir_pair.second]);
    charge.push_back(charge[dir_pair.first] + DIR[dir_pair.second] * 2);
    charge.push_back(charge[dir_pair.first] + DIR[dir_pair.second] * 3);

    updateBorder();
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
	dataSize(x * y * z * sizeof(double)),
	data(NULL),
	GPUdata(NULL),
	hasGPU(GPU){
	
	data = new double[x * y * z];

	for(int i = 0; i < x*y*z; i++) {
		data[i] = 0.0f;
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
	return Vector3i(n % size.x % size.y, n / size.x % size.y, n / size.x / size.y);
}


double CUDA::Grid3::at(int x, int y, int z) {
	if(is_in(x, 0, size.x - 1) &&
	   is_in(y, 0, size.y - 1) &&
	   is_in(z, 0, size.z - 1)) {

		return data[p2n(x, y, z)];
	}
	else
		throw CUDA::Grid3WrongCallEx(x, y, z, size.x, size.y, size.z);
}

void CUDA::Grid3::set(double num, int x, int y, int z) {
	if(is_in(x, 0, size.x - 1) &&
	   is_in(y, 0, size.y - 1) &&
	   is_in(z, 0, size.z - 1)) {

		data[p2n(x,y,z)] = num;
	}
	else
		throw CUDA::Grid3WrongCallEx(x, y, z, size.x, size.y, size.z);
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
