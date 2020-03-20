#pragma once
#include <string>
#include <iostream>

struct Parameters {
public:
    Parameters(std::string);
       
    int SIZE;
    int SIZE_Z;

    double dL;  // in meters
    double U_0; // in kV

    double EPS;
    
    int SQ_SIZE; // >0 ->  SQ_SIZE pixels per cell
                 // <0 -> -SQ_SIZE cells per pixel

    int SCREEN_X, SCREEN_Y;
};

std::ostream& operator<<(std::ostream&, const Parameters&);
