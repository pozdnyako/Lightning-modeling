#pragma once

#include "SFML/Graphics.hpp"
#include <string>
#include "Parameters.h"

class Interface {
public:
	Interface(const Parameters&, std::string);

	void run();
	void update();
	void draw();
private:

	sf::RenderWindow window;

	sf::Image image;
	sf::Texture texture;
	sf::Sprite sprite;
};