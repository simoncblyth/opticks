#pragma once
#include "curand_kernel.h"

struct scurandref
{
    unsigned long long idx ; 
    unsigned long long num ; 
    unsigned long long seed ;
    unsigned long long offset ;
    curandState*  states  ; 
};

