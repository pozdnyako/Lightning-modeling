#include <SFML/Graphics.hpp>

#include "Matrix.h"
#include "Solver.h"

int main() {
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

    sf::RenderWindow window(sf::VideoMode(200, 200), "SFML works!");
    /*sf::CircleShape shape(100.f);
    shape.setFillColor(sf::Color::Green);

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.draw(shape);
        window.display();
    }*/

    return 0;
}
