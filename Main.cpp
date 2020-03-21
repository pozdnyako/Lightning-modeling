#include <iostream>
#include <cstdio>
#include <SFML/Graphics.hpp>

#include "Tester.h"

#include "../StopwatchWin32/Stopwatch/Stopwatch.h"

int main() {
try {
    Parameters param("cfg/standard.txt");
    std::cout << param << std::endl;

    CUDA::Calculator calc(param);
    
    win32::Stopwatch watch;
    watch.Start();
    calc.calcU();
    watch.Stop();

    std::cout << watch.ElapsedMilliseconds() << "ms" << std::endl;

    Interface prog(param, "Lightning modeling");

    prog.run();
}
catch(Exception &ex) {
    std::cout << ex.what() << std::endl;
}
    return 0;
}

