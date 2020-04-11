#pragma once

#include "SFML/Graphics.hpp"

class Bitmap : public sf::Drawable {
public:
	Bitmap();
	Bitmap(sf::Vector2u);

	void updateSprite();

	void setPixelInfo(int val, int x, int y, int chan);
	int getPixelInfo(int x, int y, int chan);

	static const short R_CH = 0;
	static const short G_CH = 1;
	static const short B_CH = 2;
	static const short A_CH = 3;
private:
	void draw(sf::RenderTarget& target, sf::RenderStates states) const;

	sf::Uint8 *pixels;

	sf::Image image;
	sf::Texture texture;
	sf::Sprite sprite;

	sf::Vector2u size;
};