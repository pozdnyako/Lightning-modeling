#include <iostream>
#include <cstdio>
#include <SFML/Graphics.hpp>

#include "Calculator.h"
#include "Interface.h"

void Interface::update() {
    //std::cout << "Update" << std::endl;
}

int main() {
    Parameters param("cfg/standard.txt");
    std::cout << param << std::endl;

    Calculator calc;
    Interface prog(param, "Lightning modeling");

    prog.run();
    return 0;
}

bool addWithCudaTest() {
    int *a, *b, *c;
    int size = 10;

    a = new int[size];
    b = new int[size];
    c = new int[size];

    for(int i = 0; i < size; i++) {
        a[i] = (i + 1);
        b[i] = 10 * (i + 1);
        c[i] = 0;
    }

    Calculator::addWithCuda(c, a, b, size);

    bool result = true;
    for(int i = 0; i < size; i++) {
        if(a[i] + b[i] != c[i])
            result = false;
    }

    return result;
}
