#include <iostream>
#include <cstdio>
#include <SFML/Graphics.hpp>

#include "Calculator.h"
#include "Interface.h"

void f() {
    std::cout << "Call";
}

int main() {
    Calculator calc;
    Interface prog(200, 200, "Lightning modeling");
    
    prog.run(f);
    return 0;
}

bool CalculatorTest() {
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
