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

    OptixBuildInputCustomPrimitiveArray& getBuildInputCPA() ; 
    
};

inline OptixBuildInputCustomPrimitiveArray& BI::getBuildInputCPA()
{
#if OPTIX_VERSION == 70000
    OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.buildInput.aabbArray ;  
#elif OPTIX_VERSION > 70000
    // Hans reports that .aabbArray is not defined in OptiX 7.5 API 
    // TODO: check exactly which version this change happened at 
    OptixBuildInputCustomPrimitiveArray& buildInputCPA = bi.buildInput.customPrimitiveArray ;  
#endif
    return buildInputCPA ; 
}

