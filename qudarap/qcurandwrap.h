#pragma once
#include "curand_kernel.h"
#include "qrng.h"

template<typename T>
struct qcurandwrap
{
    unsigned long long num ; 
    unsigned long long seed ; 
    unsigned long long offset ; 
    T*                 states  ; 
};
