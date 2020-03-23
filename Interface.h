#pragma once

#include <string>
#include "Parameters.h"

#include "../StopwatchWin32/Stopwatch/Stopwatch.h"
#include "Calculator.h"
#include "Bitmap.h"

class Interface {
public:
	Interface(const Parameters&, std::string, CUDA::Calculator *);

	void run();
	void update();
	void draw();
	void updateImage();
private:
	Parameters param;
	CUDA::Calculator *calc;

	win32::Stopwatch watch;

	sf::RenderWindow window;
	
	sf::Uint8 *pixels;
	sf::Vector2u pixelsSize;

	Bitmap bitmap;

	void setPixel(int val, int x, int y, int chanal);
	int getPixelInfo(sf::Uint8* pixels, int x, int y, int chanal);
};