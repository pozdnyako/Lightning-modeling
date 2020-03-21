#include <iostream>
#include <cstdio>
#include <SFML/Graphics.hpp>

#include "Tester.h"

int main() {
try {
    Parameters param("cfg/standard.txt");
    std::cout << param << std::endl;

    CUDA::Calculator calc(param);
    calc.calcU();

    Interface prog(param, "Lightning modeling");

    prog.run();
}
catch(Exception &ex) {
    std::cout << ex.what() << std::endl;
}
    return 0;
}

