#pragma once
#include <random>
#include "SYSRAP_API_EXPORT.hh"

template <unsigned N>
struct SYSRAP_API SDice
{
    std::mt19937_64 engine ;
    std::uniform_int_distribution<unsigned>  dist ; 

    SDice(unsigned seed_=0u) : dist(0, N-1) { engine.seed(seed_); }
    unsigned operator()() { return dist(engine) ; }
};

