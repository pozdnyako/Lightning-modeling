#pragma once

#include "SFML/Graphics.hpp"
#include <string>

class Interface {
public:
	Interface(unsigned int, unsigned int, std::string);

	void run( void(*)(void) );
private:

	sf::RenderWindow window;
};