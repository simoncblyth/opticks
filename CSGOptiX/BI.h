#pragma once
#include <optix.h>

struct BI
{
    unsigned        mode ;   // 1: 11N (now the only mode)  0:1NN (original mode that failed to work)
    CUdeviceptr     d_aabb ;
    CUdeviceptr     d_sbt_index ;
    unsigned*       flags ;  
    unsigned*       sbt_index ;  

    OptixBuildInput buildInput ; 

};
