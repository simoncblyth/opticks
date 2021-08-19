#pragma once
#include <optix.h>

struct BI
{
    unsigned        mode ;   // 0: 1NN  1: 11N
    CUdeviceptr     d_aabb ;
    CUdeviceptr     d_sbt_index ;
    unsigned*       flags ;  
    unsigned*       sbt_index ;  

    OptixBuildInput buildInput ; 

};
