#include <SFML/Graphics.hpp>

#include <cmath>

#include "Matrix.h"
#include "Matrix3.h"
#include "Solver.h"

#include <iostream>
#include <vector>

struct Point {
    int x, y;

    double u;

    Point() {}
    Point(int x, int y)
    :x(x)
    ,y(y) {}
};

struct Vec {
    double x, y;

    Vec() {}
    Vec(double x, double y)
    :x(x)
    ,y(y) {}

    double sq() {
        return x*x + y*y;
    }
};


double Lambda(Matrix3 &A, int y, int x, int z, int N, int N_z) {
    double val = 0.0f;

    if(x != 0) {
        val += A.get_num(y, x - 1, z);
    }

    if(y != 0) {
        val += A.get_num(y - 1, x, z);
    }

    if(z != 0) {
        val += A.get_num(y, x, z - 1);
    }

    if(x != N - 1) {
        val += A.get_num(y, x + 1, z);
    }

    if(y != N - 1) {
        val += A.get_num(y + 1, x, z);
    }


    if(z != N_z - 1) {
        val += A.get_num(y, x, z + 1);
    }

    return val;
}

void set_u(Matrix3 &u, std::vector<Point> &point, int N, int N_z, double U_0) {
    for(int z = 0; z < N_z; z ++) {
    for(int x = 0; x < N; x ++) {
    //for(int y = 0; y < N; y ++) {
        u.set_num(0, x, z, U_0);
    }}

    for(int i = 0; i < point.size(); i ++) {
        u.set_num(point[i].y, point[i].x, N_z/2, point[i].u);
    }

    //}
}

void calc_U(Matrix3 &f, Matrix3 &u, Matrix3 &u_prev, double eps, int N, int N_z, std::vector<Point> &point, double U_0) {
    int n_op = N * N * N_z * log(1 / eps);

    printf("n operation: %d\n", n_op);

    int n_k = 1000*10;

    for(int t = 0; t < n_op / n_k /* !!!!!!!!!!!!!!!!!!!!*/ ; t ++) {

        for(int x = 0; x < N; x ++) {
        for(int y = 0; y < N; y ++) {
        for(int z = 0; z < N_z; z ++) {
            double d = -1.0f / 6.0f * Lambda(u_prev, y, x, z, N, N_z);
            u.set_num(y, x, z, d);
        }}}

        set_u(u, point, N, N_z, U_0);

        for(int x = 0; x < N; x ++) {
        for(int y = 0; y < N; y ++) {
        for(int z = 0; z < N_z; z ++) {
            u_prev.set_num(y, x, z, u.get_num(y, x, z));
        }}}

        if(t % 100 == 0)
            printf("%5.1f\%\n", (float)t * 100 * n_k / n_op);
    }
}

struct Prob{
    double p[8];

    Prob(){
        for(int i = 0; i < 8 ;i ++) {
            p[i] = 0.0f;
        }
    }
};

Vec calc_E(Point point, Matrix3 &u, int N_z) {
    return Vec((-u.get_num(point.y, point.x + 1, N_z/2) + u.get_num(point.y, point.x - 1, N_z/2))/200.0f,
               (-u.get_num(point.y + 1, point.x, N_z/2) + u.get_num(point.y - 1, point.x, N_z/2))/200.0f + 0.03);
}

void calc_prob(Point &point, Matrix3 &u, Matrix3 &f, double charge_rho, int N_z, int N, Prob *p) {
    int counter = 0;
    for(int dy = -1; dy <= 1; dy ++) {
    for(int dx = -1; dx <= 1; dx ++) {
        if(dx * dx + dy * dy >= 1) {

            p->p[counter] = 0.0f;
            if((point.x + dx * 2 > 0 && point.x + dx * 2 < N &&
                point.y + dy * 2 > 0 && point.y + dy * 2 < N) ) {

                if(std::abs(f.get_num(point.y + dy, point.x + dx, N_z/2)) < std::abs(charge_rho * 0.9) ) {

                    Vec E = calc_E(Point(point.x + dx, point.y + dy), u, N_z);

                    //std::cout << E.x << " " << E.y << std::endl;

                    double prob = pow(((E.x * (double)dx + E.y * (double)dy))/(dx*dx + dy*dy) - 0.1, 3);

                    if(prob > 0.0f) {
                        p->p[counter] = prob;
                    }
                }
            }

            counter ++;
        }
    }}

    // UP -> LEFT -> RIGHT -> DOWN
}

void draw(sf::RenderWindow &window, Matrix3 &u, Matrix3 f, std::vector<Point> &point, double charge_rho, int N, int N_z, int N_S, double U_0, sf::Image &image, sf::Texture texture, sf::Sprite sprite) {
    double u_max = 0.0, u_min = 0.0;

    for(int x = 0; x < N; x ++) {
    for(int y = 0; y < N; y ++) {

        if(u.get_num(y,x,N_z/2) > u_max) {
            u_max = u.get_num(y,x,N_z/2);
        }

        if(u.get_num(y,x,N_z/2) < u_min) {
            u_min = u.get_num(y,x,N_z/2);
        }
    }}

    std:: cout << "U in range <" << u_min << ",\t" << u_max << ">\n";

    double e_max = 0.0;

    for(int x = 1; x < N-1; x ++) {
    for(int y = 1; y < N-1; y ++) {
        double e = calc_E(Point(x,y), u, N_z).sq();

        if(e > e_max) {
            e_max = e;
        }
    }}

    std::cout << "|E| < " << e_max << "\n";


    for(int x = 0; x < N; x ++) {
    for(int y = 0; y < N; y ++) {
        for(int dx = 0; dx < N_S; dx ++) {
        for(int dy = 0; dy < N_S; dy ++) {
            //image.setPixel(x, y, sf::Color(255 * pow((u.get_num(y,x,N_z/2) - u_min) / (u_max - u_min), 0.1), 0, 0));
            image.setPixel(x*N_S+dx, y*N_S+dy, sf::Color(255 * (u.get_num(y,x,N_z/2) - u_min) / (u_max - u_min), 0, 0));

        }}
    }}

    for(int i = 0; i < point.size(); i ++) {
        for(int dx = 0; dx < N_S; dx ++) {
        for(int dy = 0; dy < N_S; dy ++) {
            image.setPixel(point[i].x * N_S + dx + N*N_S, point[i].y*N_S+dy, sf::Color(0, 255 * point[i].u / U_0, 0));
        }}
    }


    for(int x = 1; x < N-1; x ++) {
    for(int y = 1; y < N-1; y ++) {
        for(int dx = 0; dx < N_S; dx ++) {
        for(int dy = 0; dy < N_S; dy ++) {
            image.setPixel(x*N_S+dx + N * 2 * N_S, y*N_S+dy, sf::Color(0, 0, 255 * calc_E(Point(x,y), u, N_z).sq() / e_max));
        }}

    }}

    texture.loadFromImage(image);
    sprite.setTexture(texture);

    window.clear();
    window.draw(sprite);
    window.display();
}

#include "../StopwatchWin32/Stopwatch/Stopwatch.h"

int main() {
    srand(time(NULL));
    //freopen("out.txt", "wt", stdout);

    const int N = 128;
    const int N_z = 16;
    const int N_S = 3;
    const double U_0 = 80;
    const double E_i = 1 * U_0 / 1000;

    Matrix3 f(N, N, N_z);
    Matrix3 u(N, N, N_z);
    Matrix3 u_prev(N, N, N_z);

    printf("mem allocated\n");

    const double EPS = 0.1;
    const double charge = 0.01 * N * N * N_z;

    std::vector<Point> point;
    std::vector<Prob> prob;

    point.push_back(Point(N/2, 1));
    prob.push_back(Prob());

    point[0].u = U_0;

    sf::RenderWindow window(sf::VideoMode(N * N_S * 3, N * N_S), "SFML works!");

    sf::Image image;
    image.create(N * N_S * 3, N * N_S, sf::Color(0, 0, 0));

    sf::Texture texture;
    sf::Sprite sprite;

    set_u(u, point, N, N_z, U_0);
    set_u(u_prev, point, N, N_z, U_0);

    win32::Stopwatch watch;
    watch.Start();
    calc_U(f, u, u_prev, EPS, N, N_z, point, U_0);
    watch.Stop();

    std::cout << "time:" << watch.ElapsedMilliseconds() << std::endl;
    while (window.isOpen()) {

        int n_point = (int) point.size();

        printf("n points: %d\n", n_point);

        draw(window, u, f, point, charge / n_point, N, N_z, N_S, U_0, image, texture, sprite);

        /*for(int i = 0; i < n_point; i ++) {
            f.set_num(point[i].y, point[i].x, N_z/2, charge / n_point);
        }*/ 

        calc_U(f, u, u_prev, EPS, N, N_z, point, U_0);
        set_u(u, point, N, N_z, U_0);

        double prob_sum = 0.0f;

        for(int i = 0; i < n_point; i ++) {
            calc_prob(point[i], u, f, charge / n_point, N_z, N, &(prob[i]));

            for(int j = 0; j < 8; j ++) {
                prob_sum += prob[i].p[j];

                //printf("\t%f\n", prob[i].p[j]);
            }
        }
        printf("sum: %f\n", prob_sum);

        float d = ((double)rand() / (RAND_MAX - 1)) * prob_sum;

        printf("d: %f\n", d);


        for(int i = 0; i < n_point; i ++) {

            int id = -1;
            for(int j = 0; j < 8; j ++) {
                d -= prob[i].p[j];

                if(d <= 0.0f) {
                    id = j;
                    break;
                }
            }
            if(id == -1) {
                continue;
            }

            printf("ID: %d, d: %f\n", id, d);

            switch(id){
            case 0:
                printf("L-U\n");
                point.push_back(Point(point[i].x - 1, point[i].y - 1));
                prob.push_back(Prob());
                break;
            case 1:
                printf("U\n");
                point.push_back(Point(point[i].x, point[i].y - 1));
                prob.push_back(Prob());
                break;
            case 2:
                printf("R-U\n");
                point.push_back(Point(point[i].x + 1, point[i].y - 1));
                prob.push_back(Prob());
                break;
            case 3:
                printf("L\n");
                point.push_back(Point(point[i].x - 1, point[i].y));
                prob.push_back(Prob());
                break;
            case 4:
                printf("R\n");
                point.push_back(Point(point[i].x + 1, point[i].y));
                prob.push_back(Prob());
                break;
            case 5:
                printf("L-D\n");
                point.push_back(Point(point[i].x - 1, point[i].y + 1));
                prob.push_back(Prob());
                break;
            case 6:
                printf("D\n");
                point.push_back(Point(point[i].x, point[i].y + 1));
                prob.push_back(Prob());
                break;
            case 7:
                printf("DOWN\n");
                point.push_back(Point(point[i].x + 1, point[i].y + 1));
                prob.push_back(Prob());
                break;
            }

            if(id != -1) {
                printf("point u: %f\n", U_0 - E_i * n_point);
                point[n_point].u = point[i].u - E_i;
                //point[n_point+1].u = point[i].u - E_i;
            }
            break;
        }

        printf("\n\n\n");

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }
    }

    return 0;
}

/*

    Matrix f(N, N);
    Matrix u(N, N);
    Matrix u_prev(N, N);

    printf("mem allocated\n");

    f.set_num(N/2, N/2 - 50, -100);
    f.set_num(N/2, N/2 + 50, 100);

    double DELTA = 0.25f / N / N;
    double eps = 0.1;
    int n_op = 2 * N * N / 3.14 / 3.14 * log(1 / eps);

    printf("epsilon: %f\n", eps);
    printf("n operation: %d\n", n_op);

    for(int t = 0; t < n_op; t ++) {
        for(int x = 0; x < N; x ++) {
        for(int y = 0; y < N; y ++) {
            double d = 0.25 * f.get_num(y, x) / N / N + 0.25 * Lambda(u_prev, y, x, N);
            u.set_num(y, x, d);
        }}

        for(int x = 0; x < N; x ++) {
        for(int y = 0; y < N; y ++) {
            u_prev.set_num(y, x, u.get_num(y, x));
        }}

        printf("%5.1f\%\n", (float)t * 100 / n_op);
    }

    FILE* file = fopen("out.txt", "wt");

    for(int x = 0; x < N; x ++) {
        fprintf(file, "%lf\n", u.get_num(50, x));
    }

    fclose(file);

    sf::Image image;
    image.create(N, N, sf::Color(0, 0, 0));

    double u_max = 0.0, u_min = 0.0;

    for(int x = 0; x < N; x ++) {
    for(int y = 0; y < N; y ++) {
        if(u.get_num(y,x) > u_max) {
            u_max = u.get_num(y,x);
        }

        if(u.get_num(y,x) < u_min) {
            u_min = u.get_num(y,x);
        }
    }}
    printf("u_min: %f\n", u_min);
    printf("u_max: %f\n", u_max);


    double u_line = 0.000000001;

    for(int x = 0; x < N; x ++) {
    for(int y = 0; y < N; y ++) {
        image.setPixel(x, y, sf::Color(255 * ((u.get_num(y,x) - u_min) / (u_max - u_min)), 0, 0));

    }}

    for(int x = 0; x < N; x ++) {
    for(int y = 0; y < N; y ++) {
        if( std::abs((u.get_num(y,x) - u_line) / (u_line)) < 0.05 ) {
            image.setPixel(x, y, sf::Color(0, 255, 0));
        }
    }}

    sf::RenderWindow window(sf::VideoMode(N, N), "SFML works!");
    sf::Texture texture;
    texture.loadFromImage(image);

    sf::Sprite sprite;
    sprite.setTexture(texture);


    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(sprite);
        window.display();
    }
*/

/*


    Matrix A(3, 3);
    Matrix B(3, 1);

    A.set_num(0, 0, 2);
    A.set_num(0, 1, 4);
    A.set_num(0, 2, 1);

    A.set_num(1, 0, 5);
    A.set_num(1, 1, 2);
    A.set_num(1, 2, 1);

    A.set_num(2, 0, 2);
    A.set_num(2, 1, 3);
    A.set_num(2, 2, 4);

    B.set_num(0, 0, 36);
    B.set_num(1, 0, 47);
    B.set_num(2, 0, 37);

    Matrix ans(3, 1);

    solve(A, B, &ans);

    ans.print();
    A.print();
    B.print();

*/
