#pragma once

#include "NPY_API_EXPORT.hh"

struct NPY_API NSphereSDF 
{
    NSphereSDF(double x, double y, double z, double r) : x(x), y(y), z(z), r(r) {}

    double operator()(double px, double py, double pz)
    {
        return (px-x)*(px-x) + (py-y)*(py-y) + (pz-z)*(pz-z) - r*r ;
    } 
    double x ; 
    double y ; 
    double z ; 
    double r ; 
};



