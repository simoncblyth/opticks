#pragma once
#include "curand_kernel.h"   // need header as curandState is typedef to curandXORWOW

struct SYSRAP_API SCurandChunkData
{
    long idx ;     
    long num ; 
    long seed ;
    long offset ; 
    curandState* states ;
};


