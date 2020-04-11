#pragma once
#include <string>
#include <iostream>

struct Parameters {
public:
    Parameters() :SIZE(0), SIZE_Z(0), dL(0), U_0(0), EPS(0), SQ_SIZE(0) {};
    Parameters(std::string);
       
    int SIZE;
    int SIZE_Z;

    double dL;  // in meters
    double U_0; // in kV

    double EPS;
    
    int SQ_SIZE; // >0 ->  SQ_SIZE pixels per cell
                 // <0 -> -SQ_SIZE cells per pixel

    int SCREEN_X, SCREEN_Y;
    int SECTION_SIZE;
};

std::ostream& operator<<(std::ostream&, const Parameters&);
