#pragma once
#include <random>
#include "SYSRAP_API_EXPORT.hh"

template <typename T>
struct SYSRAP_API SRng
{
    std::mt19937_64 engine ;
    std::uniform_real_distribution<T>  dist ; 

    SRng(unsigned seed_=0u) : dist(0, 1) { engine.seed(seed_); }
    T operator()(){ return dist(engine) ; }
};

