#include "Interface.h"

Interface::Interface(unsigned int x_size, unsigned int y_size, std::string name)
:window(sf::VideoMode(x_size, y_size), name) {

}

void Interface::run(void (*func)(void)) {

	func();
}