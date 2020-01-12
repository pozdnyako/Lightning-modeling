#include "Solver.h"

#include <cstdio>
#include <cmath>

void solve(Matrix A, Matrix B, Matrix *ans) {
    #define A(y,x) A.get_num(y,x)
    #define B(y,x) B.get_num(y,x)
    #define ans(y,x) ans->get_num(y,x)

    if(A.get_sizeY() != B.get_sizeY() ||
       A.get_sizeY() != ans->get_sizeY() ||
       A.get_sizeX() != A.get_sizeY() ||
       B.get_sizeX() != 1 ||
       ans->get_sizeX() != 1) {

        printf("[ERROR]\t" "wrong size\n");
        return;
    }

    int n = A.get_sizeX();

    const double EPS = 0.0001;

    for(int k = 0; k < n; k ++) {
        double max = std::abs(A(k,k));

        int index = k;

        // find max element in column
        //printf("find max\n");

        for(int i = k + 1; i < n; i ++) {
            if(std::abs(A(k, i)) > max) {
                max = std::abs(A(k, i));
                index = i;
            }
        }

        if(max < EPS) {

            printf("system don't have a solution");
            return;
        }

        // string swaping
        //printf("swaping\n");

        if(index != k) {

            for(int j = 0; j < n; j ++) {
                double t = A(k, j);
                A.set_num(k, j, A(index, j));
                A.set_num(index, j, t);
            }

            double t = B(k, 0);
            B.set_num(k, 0, B(index, 0));
            B.set_num(index, 0, t);
        }

        // normalizing
        //printf("normalizing\n");

        for(int i = k; i < n; i ++) {
            double t = A(i, k);

            if(std::abs(t) < EPS)
                continue;

            for(int j = 0; j < n; j ++) {
                A.set_num(i, j, A(i,j) / t);
            }
            B.set_num(i, 0, B(i, 0) / t);

            if(i == k)
                continue;

            for(int j = 0; j < n; j ++) {
                A.set_num(i, j, A(i,j) - A(k,j));
            }
            B.set_num(i, 0, B(i,0) - B(k, 0));
        }
    }

    // solving

    for(int k = n - 1; k >= 0; k --) {
        ans->set_num(k,0, B(k,0));

        for(int i = 0; i < k; i ++) {
            B.set_num(i, 0, B(i,0) - A(i,k) * ans(k, 0));
        }
    }


    #undef A
    #undef B
    #undef ans

}
