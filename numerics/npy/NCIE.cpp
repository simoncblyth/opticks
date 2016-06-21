
#include "NCIE.hpp"
#include "ciexyz.h"


float cie_X(float nm)
{
    float X = xFit_1931(nm);
    return X ;
}

float cie_Y(float nm)
{
    float Y = yFit_1931(nm);
    return Y ;
}

float cie_Z(float nm)
{
    float Z = zFit_1931(nm);
    return Z ;
}


float NCIE::X(float nm) { return xFit_1931(nm); }
float NCIE::Y(float nm) { return yFit_1931(nm); }
float NCIE::Z(float nm) { return zFit_1931(nm); }



