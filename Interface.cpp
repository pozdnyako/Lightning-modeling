#include "Interface.h"


Interface::Interface(const Parameters &_param, std::string name, CUDA::Calculator *_calc) :
    param(_param),
    calc(_calc),
    window(sf::VideoMode(_param.SCREEN_X, _param.SCREEN_Y), name),
    bitmap(sf::Vector2u(_param.SCREEN_X, _param.SCREEN_Y)) {

}

void Interface::run() {
    watch.Start();
	while (window.isOpen()) {
        update();

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        draw();
    }
}

void Interface::update() {
    watch.Reset();
    watch.Start();

    calc->calcU();
    watch.Stop();

    std::cout << "calcU:\t" << watch.ElapsedMilliseconds() << " ms" << std::endl;

    updateImage();
}

void Interface::updateImage() {
    if(param.SQ_SIZE > 0) {
        double minU = 0;
        double maxU = 0;

        for(int x = 0; x < param.SIZE; x++) {
            for(int y = 0; y < param.SIZE; y++) {
                double U = calc->getU(x, y, param.SIZE_Z / 2);

                if(U > maxU)
                    maxU = U;

                if(U < minU)
                    minU = U;
            }
        }

        for(int x = 0; x < param.SIZE; x++) {
        for(int y = 0; y < param.SIZE; y++) {
            double U = calc->getU(x, param.SIZE - 1 - y, param.SIZE_Z / 2);

            for(int dx = 0; dx < param.SQ_SIZE; dx++) {
            for(int dy = 0; dy < param.SQ_SIZE; dy++) {
                if(U > 0 && maxU != 0.0) {
                    bitmap.setPixelInfo(255.0 * pow(U / maxU, 0.1),
                                        x*param.SQ_SIZE+dx, y*param.SQ_SIZE+dy, Bitmap::R_CH);
                    
                    bitmap.setPixelInfo(0,
                                        x*param.SQ_SIZE+dx, y*param.SQ_SIZE+dy, Bitmap::B_CH);
                }
                if(U < 0 && minU != 0.0) {
                    bitmap.setPixelInfo(255.0 * pow(U / minU, 0.1),
                                        x*param.SQ_SIZE+dx, y*param.SQ_SIZE+dy, Bitmap::B_CH);
                    
                    bitmap.setPixelInfo(0,
                                        x*param.SQ_SIZE+dx, y*param.SQ_SIZE+dy, Bitmap::R_CH);
                }

            }
            }
        }
        }
    }
    bitmap.updateSprite();
}

void Interface::draw() {

    window.clear();

    window.draw(bitmap);

    window.display();
}