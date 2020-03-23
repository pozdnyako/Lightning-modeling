#include "Bitmap.h"

Bitmap::Bitmap() :
	size(0, 0) {
	
	pixels = NULL;
}

Bitmap::Bitmap(sf::Vector2u _size) :
	size(_size) {

	image.create(_size.x, _size.y);
	pixels = (sf::Uint8*)image.getPixelsPtr();

	for(int x = 0; x < size.x; x++) {
	for(int y = 0; y < size.y; y++) {
		setPixelInfo(0, x, y, R_CH);
		setPixelInfo(0, x, y, G_CH);
		setPixelInfo(0, x, y, B_CH);
		setPixelInfo(255, x, y, A_CH);
	}}

	texture.loadFromImage(image);
	sprite.setTexture(texture);
}

void Bitmap::setPixelInfo(int val, int x, int y, int chan) {
	pixels[4 * (x + y * size.x) + chan] = val;
}

int Bitmap::getPixelInfo(int x, int y, int chan) {
	return pixels[4 * (x + y * size.x) + chan];
}

void Bitmap::updateSprite() {
	texture.loadFromImage(image);
	sprite.setTexture(texture);
}

void Bitmap::draw(sf::RenderTarget& target,sf::RenderStates states) const {
	target.draw(sprite, states);
}