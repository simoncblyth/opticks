#pragma once
#include <optix.h>

struct BI
{
    float*          param ;  // 4*4
    float*          aabb ;   // 6 
    unsigned        num_sbt_records ; 
    CUdeviceptr     d_aabb ;
    CUdeviceptr     d_sbt_index ;
    unsigned*       flags ;  
    unsigned*       sbt_index ;  

    OptixBuildInput buildInput ; 

};
