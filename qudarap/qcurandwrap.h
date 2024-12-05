#pragma once
#include "curand_kernel.h"

struct qcurandwrap
{
    unsigned long long num ; 
    unsigned long long seed ; 
    unsigned long long offset ; 
    curandState*  states  ; 
};
