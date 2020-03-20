#include "Interface.h"

Interface::Interface(const Parameters &param, std::string name)
:window(sf::VideoMode(param.SCREEN_X, param.SCREEN_Y), name){

    image.create(param.SCREEN_X, param.SCREEN_Y, sf::Color(0, 0, 0));
}

void Interface::run() {

	while (window.isOpen()) {
        update();

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear();
        window.display();
    }
}

void Interface::draw() {
    texture.loadFromImage(image);
    sprite.setTexture(texture);

    window.clear();

    window.draw(sprite);

    window.display();
}