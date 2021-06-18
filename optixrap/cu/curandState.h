#pragma once
/**
curandState : MOCKED 
=====================

Mocking the real curand for use in CPU based unit tests 
Using precooked randoms loaded from file.

**/

#include <iostream>
#include "NP.hh"

struct curandState 
{
    NP*      array  ;
    double*  u ; 
    unsigned cursor ; 
    curandState(const char* path); 
    float nextf(); 
};

curandState::curandState(const char* path)
    :
    array(NP::Load(path)),
    u(array->values<double>()),
    cursor(0)
{
    std::cout << array->desc() << std::endl ; 
}

inline float curandState::nextf()
{
    // TODO: handle reading beyond the array, eg by warning and wrapping 
    float f = float(u[cursor]); 
    cursor++ ; 
    return f ; 
}

float curand_uniform(curandState* s){ return s->nextf() ; }

