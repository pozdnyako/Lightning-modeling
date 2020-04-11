#include <iostream>
#include <cstdio>
#include <SFML/Graphics.hpp>

#include "Tester.h"

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

int main() {
try {
    CUDA::printDevicesProperties();

    Parameters param("cfg/standard.txt");
    std::cout << param << std::endl;

    CUDA::Calculator calc(param);

    Interface prog(param, "Lightning modeling", &calc);
    prog.run();
}
catch(Exception &ex) {
    std::cout << ex.what() << std::endl;
}
    return 0;
}

