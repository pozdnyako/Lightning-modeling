#include "Tester.h"

bool Test::_CUDA_addInt() {
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

    CUDA::addInt(c, a, b, size);

    bool result = true;
    for(int i = 0; i < size; i++) {
        if(a[i] + b[i] != c[i])
            result = false;
    }

    return result;
}

bool Test::_Vector() {
    Vector a(0.1, 0.2), b(0.3, 0.4);

    bool result = true;

    if((a + b).x != a.x + b.x) result = false;
    if((a + b).y != a.y + b.y) result = false;
    
    if((a - b).x != a.x - b.x) result = false;
    if((a - b).x != a.x - b.x) result = false;
    
    if((a * b) != a.x * b.x + a.y * b.y) result = false;

    return result;
}